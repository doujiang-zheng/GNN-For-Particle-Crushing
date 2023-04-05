from datetime import datetime
import logging
import os
import sys
import random
import time

import gpustat
import numpy as np
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pyg.seed_everything(seed)


def write_result(val_metrics,
                 metrics,
                 dataset,
                 params,
                 method='GTC',
                 results='results'):
    res_path = '{}/{}-{}.csv'.format(results, method, dataset)
    val_keys = val_metrics.keys()
    test_keys = metrics.keys()
    param_keys = params.keys()
    headers = ['method', 'dataset'
               ] + list(val_keys) + list(test_keys) + list(param_keys)
    if not os.path.exists(res_path):
        f = open(res_path, 'w')
        f.write(','.join(headers) + '\r\n')
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = '{},{}'.format(method, dataset)
        result_str += ',' + ','.join(
            ['{:.4f}'.format(val_metrics[k]) for k in val_keys])
        result_str += ',' + ','.join(
            ['{}'.format(metrics[k]) for k in test_keys])
        logging.info(result_str)
        params_str_list = []
        for _, p in params.items():
            p_str = ','.join([f'{k}={v}' for k, v in p.items()])
            p_str = f'"{p_str}"'
            params_str_list.append(p_str)
        # params_str = ','.join(
        #     ['{}={:.2f}'.format(k, v) for k, v in params.items()])
        # params_str = '"{}"'.format(params_str)
        params_str = ','.join(params_str_list)
        row = result_str + ',' + params_str + '\r\n'
        f.write(row)


def get_free_gpu():
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(
        lambda gpu: float(gpu.entry['memory.total']) - float(gpu.entry[
            'memory.used']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    bestGPU = max(pairs, key=lambda x: x[1])[0]
    print('setGPU: Setting GPU to: {}'.format(bestGPU))
    return str(bestGPU)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f s' % (method.__name__, te - ts))
        return result

    return timed


def set_logger(log_file=False, name='MAIN'):
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    # set up logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    if logger.hasHandlers():
        logger.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        fh = logging.FileHandler('log/{}-{}.log'.format(name, 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3, path='', min_epoch=-1):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.path = path + '_{}.pth' if path!='' else ''
        self.min_epoch = min_epoch

    def early_stop_check(self, curr_val, model):
        if not self.higher_better:
            curr_val *= -1

        if self.last_best is None:
            self.last_best = curr_val
            torch.save(model.state_dict(), self.path.format(self.epoch_count))

        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            torch.save(model.state_dict(), self.path.format(self.epoch_count))

        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round and self.epoch_count > self.min_epoch
    
    def get_best_model(self, model):
        best_model_path = self.path.format(self.best_epoch)
        model.load_state_dict(torch.load(best_model_path))
        return model


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


class Displayer(object):
    def __init__(self, num_data=2, sort_id=-1, legend=[''], xlabel='', ylabel='', title='' ):
        self.num_data = num_data
        self.y = [[] for _ in range(self.num_data)]
        self.sort_id = sort_id
        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
    
    def record(self, list):
        '''list=[np.array(n),np.array(n),...]'''
        if self.y[0] == []:
            for i, data in enumerate(list):
                if isinstance(data, float):
                    data = np.array([data])
                elif not isinstance(data, np.ndarray):
                    data = np.array(data)
                self.y[i] = data
        else:
            for i, data in enumerate(list):
                if isinstance(data, float):
                    data = np.array([data])
                elif not isinstance(data, np.ndarray):
                    data = np.array(data)
                self.y[i] = np.append(self.y[i], data, axis=0)
    
    def plt(self, mode='scatter', show=0, save_path='', title=''):
        plt.figure()
        x = np.arange(len(self.y[0]))
        if self.sort_id>=0:
            idx = np.argsort(self.y[self.sort_id])
        # cmp = plt.cm.get_cmap('hsv', self.num_data)
        colors = 'rbgcmykw'
        for i, data in enumerate(self.y):
            if self.sort_id>=0:
                data = data[idx]
            if mode=='plot':
                plt.plot(x, data, c=colors[i])
            else:
                plt.scatter(x, data, c=colors[i])
        plt.legend(self.legend)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if title!='': self.title = title
        plt.title(self.title)
        if save_path!='':
            plt.savefig(save_path)
        if show:
            plt.show()
    
    def transform(self, scaler):
        for i, data in enumerate(self.y):
            scaled = scaler.inverse_transform(data[:,None])
            self.y[i] = np.squeeze(scaled, 1)
            