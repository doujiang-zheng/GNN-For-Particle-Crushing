import multiprocessing
import random

import easydict
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pyximport
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_util import load_data

pyximport.install(setup_args={'include_dirs': np.get_include()})
import pickle

import algos

from collator_graphormer import (Batch, pad_1d_unsqueeze, pad_2d_unsqueeze,
                                 pad_3d_unsqueeze, pad_attn_bias_unsqueeze,
                                 pad_edge_type_unsqueeze,
                                 pad_spatial_pos_unsqueeze)


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_graph(data):
    gfeat, graph = data
    edge_attr, edge_index, x = graph.edge_attr, graph.edge_index, graph.x
    N = x.size(0)
    # x = convert_to_single_emb(x, 8)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.float)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr #convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.detach().numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item = EasyDict({})
    item.x = x # [n, nfeat_dim]
    item.adj = adj # [n, n]
    item.attn_bias = attn_bias # [n+1, n+1]
    item.attn_edge_type = attn_edge_type # [n+1, n+1, efeat_dim]
    item.spatial_pos = spatial_pos # [n, n]
    item.in_degree = adj.long().sum(dim=1).view(-1) # [n]
    item.out_degree = adj.long().sum(dim=0).view(-1) # [n]
    item.edge_input = torch.from_numpy(edge_input) # [n, n, max_dist, efeat_dim]
    item.y = torch.FloatTensor(graph.y)
    # print(x.shape, adj.shape, attn_bias.shape, spatial_pos.shape, item.in_degree.shape, item.edge_input.shape)
    return gfeat, item

def preprocess_data(data):
    # n_jobs = multiprocessing.cpu_count() // 2
    # ans = Parallel(n_jobs=n_jobs, verbose=10)(delayed(preprocess_graph)(item) for item in data)
    ans = []
    for d in tqdm(data):
        item = preprocess_graph(d)
        ans.append(item)
    return ans

def collator(items, max_node=216, multi_hop_max_dist=8, spatial_pos_max=8):
    gfeats = torch.vstack([x for (x,y) in items])
    items = [y for (x,y) in items]
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y) for item in items]
    attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.stack(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])
    # print(x.shape, attn_bias.shape, attn_edge_type.shape, spatial_pos.shape, in_degree.shape, edge_input.shape, y.shape)
    return gfeats, Batch(
        attn_bias=attn_bias, # [n_graph, n_node+1, n_node+1]
        attn_edge_type=attn_edge_type, # [n_graph, n_node+1, n_node+1, efeat_dim]
        spatial_pos=spatial_pos, # [n_graph, n_node, n_node]
        in_degree=in_degree, # [n_graph, n_node]
        out_degree=out_degree,
        x=x, # [n_graph, n_node, node_dim]
        edge_input=edge_input, #[b, n, n, max_dist, efeat_dim]
        y=y,
    )

def load_graphormer(args, data_args, logger, model_mode='inner'):
    trainloader, validloader, testloader, gfeat_dim, nfeat_dim, efeat_dim = \
        load_data(args, data_args, logger, model_mode=model_mode)
    
    logger.info('Preprocessing Graphormer features with positional embeddings.')
    train_data = preprocess_data(trainloader.dataset)
    valid_data = preprocess_data(validloader.dataset)
    test_data = preprocess_data(testloader.dataset)

    _train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    _valid = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    _test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    return _train, _valid, _test, gfeat_dim, nfeat_dim, efeat_dim
