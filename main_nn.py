from collections.abc import Iterable
import os
import json
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from easydict import EasyDict
from scipy.linalg import block_diag
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch_geometric.nn import MLP, GIN
from torch_scatter import scatter
from tqdm import trange, tqdm

from args_config import particle_args, train_args
from data_graphormer import load_graphormer
from data_meshnet import load_meshnet
from data_util import load_data
from expc.model import EXPC
from graphormer.model import GraphFormer
from meshnet.MeshNet import MeshNet
from model_config import get_model_config
from util import (Displayer, EarlyStopMonitor, get_free_gpu, set_logger, set_random_seed,
                  write_result)

# from tensorboardX import SummaryWriter
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

class GlueModel(nn.Module):
    '''We wrap a GlueModel around the various GNN models to switch on or off different features.
    '''

    def __init__(self, args, model_args) -> None:
        super().__init__()

        gfeat_dim = model_args.gfeat_dim
        nfeat_dim = model_args.nfeat_dim
        efeat_dim = model_args.efeat_dim
        hidden = model_args.hidden
        output = model_args.output
        dropout = model_args.dropout

        self.model_name = args.model
        if args.model == 'MLP':
            channel_list = [gfeat_dim] + [hidden] * model_args.layers + [output]
            self.model = MLP(channel_list, dropout=dropout)
        elif args.model == 'MeshNet':
            self.model = MeshNet(model_args)
        elif args.model == 'GIN':
            self.model = GIN(nfeat_dim, hidden, model_args['layers'], output, \
                        model_args['dropout'], jk=model_args['JK'])
        elif args.model == 'Graphormer':
            self.model = GraphFormer(model_args, nfeat_dim, efeat_dim, output)
        elif args.model == 'ExpC':
            self.model = EXPC(model_args, nfeat_dim, efeat_dim, output)
        else:
            raise NotImplementedError(args.model)
        
        self.using_gfeat = args.using_gfeat
        self.remove_nfeat = args.remove_nfeat
        self.remove_efeat = args.remove_efeat
        self.gfeat_mlp = nn.Sequential(
            nn.Linear(gfeat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output))
    
    def forward(self, gfeat, graph):
        if self.model_name == 'MLP':
            outputs = self.model(gfeat)
        elif self.model_name == 'GIN':
            #  g_ = Data(x=nfeat, edge_index=eidx, edge_attr=efeat, y=y)
            x, eidx, batch = graph.x, graph.edge_index, graph.batch
            node_outputs = self.model(x, eidx).squeeze(1)
            outputs = scatter(node_outputs, batch, reduce='max').unsqueeze(1)
        else:
            outputs = self.model(graph)

        if self.using_gfeat:
            y_hat = self.gfeat_mlp(gfeat)
            outputs += y_hat

        return outputs
    

@torch.no_grad()
def evalModel(args, model, loader, criterion):
    model.eval()
    displayer = Displayer(num_data=2, legend=["Predict Force","True Force"], sort_id=1)
    total, total_loss = 0, 0
    y, y_hat = [], []
    for (gfeat, graph) in loader:
        gfeat = gfeat.to(args.device)
        graph = graph.to(args.device)
        outputs = model(gfeat, graph) #[b, 1]

        outputs = outputs.squeeze(1) #[b]
        label = graph.y
        loss = criterion(outputs, label)
        total += len(label)
        total_loss += loss.item() * len(label)

        outs = outputs.cpu().detach().numpy()
        lb = label.cpu().detach().numpy()
        y.append(lb)
        y_hat.append(outs)
        # displayer.record([np.squeeze(outs,1), np.squeeze(lb,1)])
        displayer.record([outs, lb])
    
    total_loss /= total
    Y = np.concatenate(y)
    Y_hat = np.concatenate(y_hat)
    return total_loss, displayer, Y, Y_hat


def trainModel(args, model, trainloader, validloader, criterion, optimizer, scheduler, early_stopper):
    loss_displayer = Displayer(num_data=2, legend=["trian loss", "valid loss"], \
            xlabel='Epoch', ylabel='loss')
    # writer = SummaryWriter('runs/mlp')

    outer = trange(args.max_epoch)
    for epoch in outer:
        model.train()
        total, train_loss = 0, 0
        for (gfeat, graph) in tqdm(trainloader):
            gfeat = gfeat.to(args.device)
            graph = graph.to(args.device)
            
            outputs = model(gfeat, graph)
            outputs = outputs.squeeze(1)
            label = graph.y

            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += len(label)
            train_loss += loss.item() * len(label)
        train_loss /= total
        valid_loss, _, _, _ = evalModel(args, model, validloader, criterion)
        scheduler.step()

        loss_displayer.record([train_loss, valid_loss])
        # writer.add_scalar(tag='valid_loss', scalar_value=valid_loss, global_step=epoch)
        # writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=epoch)
        
        outer.set_postfix(train_loss=train_loss, valid_loss=valid_loss)
        if early_stopper.early_stop_check(valid_loss, model):
            break
    print(f"Load the best model at epoch {early_stopper.best_epoch}")
    model = early_stopper.get_best_model(model)
    return model, train_loss, loss_displayer


def test_metrics(y_true, y_hat, prefix=''):
    mae = MAE(y_true, y_hat)
    rmse = np.sqrt(MSE(y_true, y_hat))
    r2 = r2_score(y_true, y_hat)
    return {f'{prefix}MAE': mae, f'{prefix}RMSE': rmse, f'{prefix}R2': r2}

    
def main(args, data_args, model_args):
    logger = set_logger(log_file=True, name=args.model)
    set_random_seed(args.seed)

    
    args.device = torch.device(f'cuda:{args.gpu_id}')
    # args.device = torch.device('cpu')

    if args.model == 'MeshNet':
        model_args['structural_descriptor'] = {'num_kernel': 64, 'sigma':0.2}
        model_args['mesh_convolution'] = {'aggregation_method': 'Max'}
        trainloader, validloader, testloader, gfeat_dim, nfeat_dim, efeat_dim = \
            load_meshnet(args, data_args, logger, model_mode=args.model_mode)
    elif args.model == 'Graphormer':
        trainloader, validloader, testloader, gfeat_dim, nfeat_dim, efeat_dim = \
            load_graphormer(args, data_args, logger, model_mode=args.model_mode)
    else:
        trainloader, validloader, testloader, gfeat_dim, nfeat_dim, efeat_dim = \
            load_data(args, data_args, logger, model_mode=args.model_mode)

    model_args = EasyDict(model_args)    
    model_args.gfeat_dim = gfeat_dim
    model_args.nfeat_dim = nfeat_dim
    model_args.efeat_dim = efeat_dim

    logger.info(args)
    logger.info(data_args)
    logger.info(model_args)

    TRAIN_PARAM_STR = f'{args.task}-{args.seed}-{args.test_choice}-{args.rotate}-{args.batch_size}-{args.max_epoch}-{args.lr:.5f}'
    FEATURE_STR = f'{args.model_mode}-Gravity{args.without_gravity}-Diameter{args.using_gfeat}-ReNode{args.remove_nfeat}-ReEdge{args.remove_efeat}'
    MODEL_PARAM_STR = '-'.join([f'{k}{v}' for k, v in model_args.items() if not isinstance(v, Iterable)])
    DEVICE_STR = f'HOST{socket.gethostname()}-GPU{args.gpu_id}'
    MODEL_NAME = f'{args.model}-{TRAIN_PARAM_STR}-{FEATURE_STR}-{MODEL_PARAM_STR}-{DEVICE_STR}'
    # MODEL_NAME = f'{args.model}-{TRAIN_PARAM_STR}-{FEATURE_STR}-{DEVICE_STR}'
    CKPT_PATH = f'./saved_checkpoints/{MODEL_NAME}'
    MODEL_PATH = f'./saved_models/{MODEL_NAME}.pth'
    early_stopper = EarlyStopMonitor(max_round=5, higher_better=False, path=CKPT_PATH, min_epoch=args.min_epoch)

    # model, config = getModel(args, nfeat_dim, efeat_dim, gfeat_dim)
    model = GlueModel(args, model_args)
    model = model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.2)
    L1Loss = nn.L1Loss()
    print(model)

    model, train_loss, loss_displayer = trainModel(args, model, trainloader, \
                                validloader, criterion, optimizer, scheduler, early_stopper)
        
    train_loss, displayer, _, _ = evalModel(args, model, trainloader, L1Loss)
    valid_loss, displayer, valid_y, valid_y_hat = evalModel(args, model, validloader, L1Loss)
    test_loss, displayer, test_y, test_y_hat = evalModel(args, model, testloader, L1Loss)

    pred_force = torch.tensor(displayer.y[0]).float()
    true_force = torch.tensor(displayer.y[1]).float()
    r2 = r2_score(true_force, pred_force)
    print(f'train_loss:{train_loss:.4f}, valid_loss:{valid_loss:.4f}, test_loss:{test_loss:.4f}, R2:{r2:.4f}')

    logger.info('Save best model at Epoch %d.', early_stopper.best_epoch)
    logger.info('MODEL PATH: %s.', MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH)

    logger.info('Write result.')
    val_metrics = test_metrics(valid_y, valid_y_hat, prefix='VAL_')
    tes_metrics = test_metrics(test_y, test_y_hat)
    DATA = data_args.file_dir_postfix
    TRAIN_PARAMS = {
        'task': args.task,
        'seed': args.seed,
        'test_choice': args.test_choice,
        'rotate': args.rotate,
        'batch_size': args.batch_size,
        'max_epoch': args.max_epoch,
        'lr': args.lr
    }
    FEATURE_PARAMS = {
        'model_mode': args.model_mode,
        'without_gravity': args.without_gravity,
        'using_gfeat': args.using_gfeat,
        'using_gdist': args.using_gdist,
        'using_ndist': args.using_ndist,
        'remove_nfeat': args.remove_nfeat,
        'remove_efeat': args.remove_efeat
    }
    MODEL_PARAMS = model_args
    params = {
        'train_params': TRAIN_PARAMS,
        'feature_params': FEATURE_PARAMS,
        'model_params': MODEL_PARAMS
    }
    write_result(val_metrics, tes_metrics, DATA, params=params, method=args.model)


if __name__ == '__main__':
    args = train_args().parse_args()
    data_args = particle_args()
    model_args = next(iter(get_model_config(args.model)))
    
    args.gpu_id = get_free_gpu() 
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    main(args, data_args, model_args)
