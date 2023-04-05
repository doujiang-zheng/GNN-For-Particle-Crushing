from ast import Param
from easydict import EasyDict

from sklearn.model_selection import ParameterGrid


def get_model_config(model_name):
    if model_name == 'Linear':
        return Linear_config()
    elif model_name == 'Ridge':
        return Ridge_config()
    elif model_name == 'RF':
        return RF_config()
    elif model_name == 'XGB':
        return XGB_config()
    elif model_name == 'LGB':
        return LGB_config()
    elif model_name == 'MLP':
        return MLP_config()
    elif model_name == 'MeshNet':
        return MeshNet_config()
    elif model_name == 'GIN':
        return GIN_config()
    elif model_name == 'ExpC':
        return ExpC_config()
    elif model_name == 'Graphormer':
        return Graphormer_config()
    else:
        return NotImplementedError(model_name)


class ParameterList:
    def __init__(self, model_args, basic_args) -> None:
        self.iter_list = []
        for k, v in model_args.items():
            self.iter_list.extend([(k, val) for val in v])
        self.basic_args = basic_args

    def __len__(self):
        return len(self.iter_list)
    
    def __iter__(self):
        yield self.basic_args # Add a default config.
        for k, v in self.iter_list:
            args = self.basic_args.copy()
            args[k] = v
            yield args


def Linear_config():
    args = {}
    return ParameterGrid(args)


def Ridge_config():
    args = {'alpha': [1e-3, 1e-2, 1e-1, 1.0]}
    return ParameterGrid(args)


def RF_config():
    args = {
        'n_estimators': [10, 25, 50, 75, 100],
        'criterion': ['squared_error', 'absolute_error', 'poisson'],
        'min_samples_split': [2, 8, 32]
    }
    return ParameterGrid(args)


def XGB_config():
    args = {
        'n_estimators': [10, 25, 50, 75, 100],
        'max_depth': [4, 16, 64],
        'reg_alpha': [0.0, 1e-3, 1e-2, 1e-1, 1.0],
        'reg_lambda': [0.0, 1e-3, 1e-2, 1e-1, 1.0],
    }
    return ParameterGrid(args)


def LGB_config():
    args = {
        'n_estimators': [10, 25, 50, 75, 100],
        'max_depth': [4, 16, 64],
        'reg_alpha': [0.0, 1e-3, 1e-2, 1e-1, 1.0],
        'reg_lambda': [0.0, 1e-3, 1e-2, 1e-1, 1.0],
    }
    return ParameterGrid(args)


def MLP_config():
    args = {
        'batch_size': [8, 32, 128, 512],
        'lr': [1e-5, 1e-4, 1e-3, 1e-2],
        'hidden': [128, 256, 512],
        'output': [1],
        'layers': [2, 3, 4, 5],
    }
    basic = {'batch_size': 128, 'lr': 1e-3, 'hidden': 128, 'output': 1, 'layers': 2, 'dropout': 0.1}
    return ParameterList(args, basic)


def MeshNet_config():
    args = {
        'batch_size': [8, 32, 128, 512],
        'lr': [1e-5, 1e-4, 1e-3, 1e-2],
        'hidden': [128, 256, 512],
        'output': [1],
        'layers': [2, 3, 4, 5],
    }
    basic = {'batch_size': 128, 'lr': 1e-3, 'hidden': 128, 'output': 1, 'layers': 2, 'dropout': 0.1}
    return ParameterList(args, basic)


def GIN_config():
    args = {
        'batch_size': [8, 32, 128, 512],
        'lr': [1e-5, 1e-4, 1e-3, 1e-2],
        'hidden': [128, 256, 512],
        'layers': [2, 3, 4, 5],
        'JK': ['last', 'cat', 'max', 'lstm'],
    }
    basic = {'batch_size': 128, 'lr': 1e-3, 'hidden': 128, 'output': 1, 'layers': 2, 'dropout': 0.1, 'eps': 1e-5, 'JK': 'max'}
    return ParameterList(args, basic)


def ExpC_config():
    args = {
        'batch_size': [8, 32, 128, 512],
        'lr': [1e-5, 1e-4, 1e-3, 1e-2],
        'hidden': [128, 256, 512],
        'layers': [2, 3, 4, 5]
    }
    basic = {'batch_size': 128, 'lr': 1e-3, 'hidden': 128, 'output': 1, 'layers': 2, 'dropout': 0.1, 'JK': 'M', 'pooling': 'M', 'exp_nonlinear': 'ELU', 'exp_n': 2, 'exp_bn': 'Y'}
    return ParameterList(args, basic)


def Graphormer_config():
    args = {
        'batch_size': [8, 32, 128, 512],
        'lr': [1e-5, 1e-4, 1e-3, 1e-2],
        'hidden': [128, 256, 512],
        'layers': [2],
    }
    basic = {'batch_size': 128, 'lr': 1e-3, 'hidden': 128, 'output': 1, 'layers': 2, 'dropout': 0.1, 'head_size': 32, 'weight_decay': 1e-5, 'warmup_updates': 60_000, 'tot_updates': 1_000_000, 'peak_lr': 2e-4, 'end_lr': 1e-9, 'edge_type': 'multi_Hop', 'multi_hop_max_dist': 5}
    return ParameterList(args, basic)
