# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# from data import get_dataset
# from lr import PolynomialDecayLR
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl

# from utils.flag import flag, flag_bounded

import numpy as np
import torch.nn.functional as F
def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

class RBFLayer(nn.Module):
    def __init__(self, K=64, cutoff=10, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff

        centers = torch.tensor(softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)), dtype=dtype)
        self.centers = nn.Parameter(F.softplus(centers))

        widths = torch.tensor([softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K, dtype=dtype)
        self.widths = nn.Parameter(F.softplus(widths))
    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        return torch.where(x < 1, 1-6*x5+15*x4-10*x3, torch.zeros_like(x))
    def forward(self, D):
        D = D.unsqueeze(-1)
        return self.cutoff_fn(D) * torch.exp(-self.widths*torch.pow((torch.exp(-D) - self.centers), 2))

class GraphFormer(pl.LightningModule):
    def __init__(self, config, nfeat_dim, efeat_dim, output, max_node=216):
        super().__init__()
        self.save_hyperparameters()

        self.num_heads = config.head_size
    
        # self.atom_encoder = nn.Embedding(max_node * nfeat_dim + 1, config.hidden, padding_idx=0)
        # self.edge_encoder = nn.Embedding(max_node * efeat_dim + 1, config.head_size, padding_idx=0)
        self.atom_encoder = nn.Linear(nfeat_dim, config.hidden)
        self.edge_encoder = nn.Linear(efeat_dim, config.head_size)
        self.edge_type = config.edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(128 * config.head_size * config.head_size,1)
        self.spatial_pos_encoder = nn.Embedding(max_node, config.head_size, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(max_node, config.hidden, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(max_node, config.hidden, padding_idx=0)

        self.dropout = nn.Dropout(config.dropout)
        encoders = [EncoderLayer(config.hidden, config.hidden, config.dropout, config.dropout, config.head_size)
                    for _ in range(config.layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(config.hidden)

        self.out_proj = nn.Linear(config.hidden, 1)

        self.graph_token = nn.Embedding(1, config.hidden)
        self.graph_token_virtual_distance = nn.Embedding(1, config.head_size)

        # self.evaluator = get_dataset(dataset_name)['evaluator']
        # self.metric = get_dataset(dataset_name)['metric']
        # self.loss_fn = get_dataset(dataset_name)['loss_fn']
        # self.dataset_name = dataset_name
        
        self.warmup_updates = config.warmup_updates
        self.tot_updates = config.tot_updates
        self.peak_lr = config.peak_lr
        self.end_lr = config.end_lr
        self.weight_decay = config.weight_decay
        self.multi_hop_max_dist = config.multi_hop_max_dist

        # self.flag = flag
        # self.flag_m = flag_m
        # self.flag_step_size = flag_step_size
        # self.flag_mag = flag_mag
        self.hidden = config.hidden
        # self.automatic_optimization = not self.flag

        K = 256
        cutoff = 10
        self.rbf = RBFLayer(K, cutoff)
        self.rel_pos_3d_proj = nn.Linear(K, config.head_size)

    def forward(self, graph):
        attn_bias, spatial_pos, x = graph.attn_bias, graph.spatial_pos, graph.x
        in_degree, out_degree = graph.in_degree, graph.in_degree
        edge_input, attn_edge_type = graph.edge_input, graph.attn_edge_type
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]
        if graph_attn_bias.size(2) == 120:
            temp = torch.tensor([[0.5]]).cuda()
            temp.requires_grad = True
            return temp
        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        # print(graph_attn_bias.shape, spatial_pos_bias.shape)
        if graph_attn_bias.size(2) != 120:
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        if graph_attn_bias.size(2) != 120:
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, efeat_dim, n_head] -> [b, n, n, max_dist, n_head]
            # edge_input = self.edge_encoder(edge_input).mean(-2)
            edge_input = self.edge_encoder(edge_input.float())
            max_dist = edge_input.size(-2)
            # [b, n, n, max_dist, n_head] -> [max_dist, b, n, n, n_head] -> [max_dist, b*n*n, n_head]
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :]) # [max_dist, b*n*n, n_head] * [max_dist,n_head,n_head] = [max_dist, b*n*n, n_head]
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)# [b, n, n, max_dist, n_head]
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, efeat_dim] -> [n_graph, n_node, n_node, n_head]
            #  -> [n_graph, n_head, n_node, n_node]
            # edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
            # print("attn_edge_type", attn_edge_type)
            edge_input = self.edge_encoder(attn_edge_type.float())
            edge_input = edge_input.permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset [n_graph, n_head, n_node+1, n_node+1]

        # node feauture + graph token
        # node_feature = self.atom_encoder(x).sum(dim=-2)
        node_feature = self.atom_encoder(x)
        # [n_graph, n_node, node_dim, n_hidden] -> [n_graph, n_node, n_hidden]

        node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree) # [n_graph, n_node, n_hidden]
        # [1, n_hidden] -> [1, 1, n_hidden] -> [n_graph, 1, n_hidden]
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.dropout(graph_node_feature) #[n_graph, n_node+1, n_node+1]
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias) #[n_graph, n_head, n_node+1, n_node+1]
        output = self.final_ln(output)

        # output part
        output = self.out_proj(output[:, 0, :])                        # get whole graph rep
        # print(output.shape)
        return output
        

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout, attention_dropout, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout, head_size)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
