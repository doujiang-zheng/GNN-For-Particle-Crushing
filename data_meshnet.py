import os
import pickle
from easydict import EasyDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_util import load_data

def pad_2d_unsqueeze(x, padlen):
    x = x+1
    xdim, xlen = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([xdim, padlen], dtype=x.dtype)
        new_x[:, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze2(x, padlen):
    # x = x+1
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

class Batch():
    def __init__(self, **kwargs):
        self.attrs = kwargs.keys()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        for k in self.attrs:
            v = getattr(self, k)
            setattr(self, k, v.to(device))
        return self

    def __len__(self):
        return self.center.size(0)

def preprocess_data(graph_data):
    trans_data = []
    # max_node_num = max([len(g.x) for _, g in graph_data])
    for gfeat, graph in tqdm(graph_data):
        node_feat, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        center = torch.transpose(node_feat[:,-3:], 0, 1) #[3, n]
        normal = torch.transpose(node_feat[:,1:4], 0, 1) #[3, n]
        num_nodes = len(node_feat)
        corner = torch.zeros((num_nodes, 9))
        # corner[:,0:1] = node_feat[:,0:1]
        # corner[:,1:3] = node_feat[:,4:6]
        ngh_index = torch.zeros((num_nodes, 3))
        for j in range(num_nodes):
            ind = torch.where(edge_index[0]==j)[0]
            ngh = edge_index[1][ind][:3]
            ngh_index[j, :len(ngh)] = ngh
            ngh_center = edge_attr[ngh, -3:] #[3, 3]
            corner[j, :3*len(ngh)] = (ngh_center - center[:, j]).flatten()

        corner = torch.transpose(corner, 0, 1)
        graph.center = center
        graph.normal = normal
        graph.corner = corner
        graph.neighbor_index = ngh_index
        trans_data.append((gfeat, graph))
    return trans_data

def collator(items):
    gfeats = torch.vstack([x for x, y in items])
    graphs = [y for x, y in items]

    max_node_num = max(g.center.size(1) for g in graphs)
    centers = torch.cat([pad_2d_unsqueeze(g.center, max_node_num) for g in graphs])
    corners = torch.cat([pad_2d_unsqueeze(g.corner, max_node_num) for g in graphs])
    normals = torch.cat([pad_2d_unsqueeze(g.normal, max_node_num) for g in graphs])
    ngh_indices = torch.cat([pad_2d_unsqueeze2(g.neighbor_index, max_node_num) for g in graphs])
    y = torch.stack([g.y for g in graphs])

    return gfeats, Batch(center=centers, corner=corners, normal=normals, neighbor_index=ngh_indices.long(), y=y)

def load_meshnet(args, data_args, logger, model_mode='inner'):
    trainloader, validloader, testloader, gfeat_dim, nfeat_dim, efeat_dim = \
        load_data(args, data_args, logger, model_mode=model_mode)
    
    logger.info('Preprocessing MeshNet features with four new features for each particle: center, normal, corner and neighbor_index.')
    train_data = preprocess_data(trainloader.dataset)
    valid_data = preprocess_data(validloader.dataset)
    test_data = preprocess_data(testloader.dataset)
    
    _train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    _valid = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    _test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    return _train, _valid, _test, gfeat_dim, nfeat_dim, efeat_dim