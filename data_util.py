import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from scipy.linalg import block_diag
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# from util import set_random_seed

def load_datapath(file_postfix, mode='inner'):
    '''
    Params:
    ----------
    file_postfix: Configurations when using the particle_generator.py.
    mode: Whether use the inner cells or the outer faces as the graph nodes.
    '''
    root_dir = 'data_files/result'
    pkl_path = f'{root_dir}/graph_{mode}_{file_postfix}.pkl'
    dat_path = f'{root_dir}/dat_{file_postfix}.csv'
    feat_path = f'{root_dir}/feature_{file_postfix}.csv'
    dist_path = f'{root_dir}/dist_{file_postfix}.npy'
    return pkl_path, dat_path, feat_path, dist_path

def weibull_fit(chr_strength):
    chr_strength = chr_strength[chr_strength > 0].to_numpy()
    chr_strength = np.sort(chr_strength)[::-1]
    n = len(chr_strength)
    sur_prob = np.arange(1, n + 1) / (n + 1)
    if len(chr_strength) <= 0:
        return 0.0, 0.0

    eps = 1e-7 # ensure the probability is larger than 0 to avoid numeric underflow
    chr_strength += eps
    sur_prob += eps
    
    ln_strength = np.log(chr_strength).reshape((-1, 1))
    mid_prob = np.clip(- np.log(sur_prob), eps, None)
    ln_prob = np.log(mid_prob).reshape((-1, 1))
    ans = np.linalg.lstsq(ln_strength, ln_prob, rcond=None)[0]
    m = ans[0, 0]
    # r = r2_score(sur_prob, np.exp(-chr_strength**m))
    
    return m, 0.0

def compute_sigma(df, rotate=True):
    '''For a specific particle type grouped by (diameter, scalex, sclaey, scalez, axisx, axisy, 
    axisz), we filter its failure simulations, and filter this particle type if its doesn't have
    more than 30 successful simulations. Then, we compute the characteristic crushing strength and
    its Weibull modulus.

    Params:
    ----------
    df: pandas.DataFrame, recording particle features in dat_{postfix}.csv.
    rotate: Whether use 3 axis rotation particles or only rotate around the axis Z.
    '''
    quant = 1 - (1 / np.e)
    cols = ['neper_diameter', 'neper_scalex', 'neper_scaley', 'neper_scalez', 'neper_axisx', 'neper_axisy', 'neper_axisz']
    type2sigma = dict()
    df['index'] = df.index
    ans = []
    for name, group in df.groupby(cols):
        # The diameter is the displacement between the two planes of the simulation.
        d0, sx, sy, sz, ax, ay, az = name
        # There is only one 1 in (ax, ay, az), determining which axis will be rotated to the vertical axis.
        # We use the (ax, ay, az) as the mask to filter the proper scale coefficient of the vertical axis.
        d = d0 * (ax*sy + ay*sx + az*sz)
        y = group['y']
        sigma = group['y'] / d**2
        if sum(sigma > 0) == 0:
            sigma0 = 0.0
            modulus = 0.0
        else:
            sigma0 = np.quantile(sigma[sigma > 0], quant)
            y0 = np.quantile(y[sigma > 0], quant)
            modulus, _ = weibull_fit(y / y0)
        type_name = '-'.join(['{:.4f}'.format(c) for c in name])
        type2sigma[type_name] = sigma0

        group['sigma0'] = sigma0
        group['modulus'] = modulus
        group['type'] = type_name
        # We ensure that the successful simulations are larger than 30.
        group['mask'] = sum(group['y'] > 0) >= 30
        ans.append(group)
    
    new_df = pd.concat(ans).sort_values(by='index')
    axisz_mask = new_df['neper_axisz'] == 1
    new_df['mask'] = new_df['mask'] & (new_df['y'] > 0)
    if not rotate:
        new_df['mask'] &= axisz_mask

    return new_df

def train_test_split(args, data_args, df, choice='diameter'):
    '''
    Params:
    ----------
    df: pandas.DataFrame, recording particle features in dat_{postfix}.csv.
    data_args: Configuration of the particle generation process.
    choice: diameter, scale or rotation, splitting the test set according to the given dimension.
    '''
    ds = df['neper_diameter'].unique()
    scale_choices = data_args.scale_choices
    rotations = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    valid_ratio = args.valid_ratio # 0.2
    test_ratio = args.test_ratio # 0.37
    if choice == 'diameter':
        num = int(len(ds) * test_ratio)
        test_ds = ds[-num:]
        test_mask = df['neper_diameter'].isin(test_ds)
    elif choice == 'scale':
        test_scales = scale_choices[1::3]
        test_xs = [s[0] for s in test_scales]
        test_ys = [s[1] for s in test_scales]
        test_mask = df['neper_scalex'].isin(test_xs) & df['neper_scaley'].isin(test_ys)
    elif choice == 'rotation':
        test_rotations = [(0, 1, 0)]
        test_mask = df['neper_axisy'] == 1
    else:
        raise NotImplementedError(choice)
    
    train_mask = np.logical_not(test_mask)
    # nn_mask = np.logical_not(test_mask)
    # indices = np.arange(len(df))[nn_mask]

    # set_random_seed()
    # np.random.shuffle(indices)
    indices = np.arange(len(df))[test_mask]
    random_state = np.random.RandomState(seed=42)
    random_state.shuffle(indices)
    num_valid = int(len(indices) * valid_ratio)
    valid_indices = indices[:num_valid]
    test_indices = indices[num_valid:]
    valid_mask = np.zeros(len(df), dtype=bool)
    valid_mask[valid_indices] = True
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_indices] = True
    # train_mask = np.zeros(len(df), dtype=bool)
    # train_mask[train_indices] = True

    # test_mask = test_mask.to_numpy()
    assert np.all(train_mask + valid_mask + test_mask)

    return train_mask, valid_mask, test_mask

def get_graph_feat(feature_df, graph_dists, using_gdist=True):
    fdf = feature_df.drop('file_id', axis=1)
    if using_gdist:
        dist = pd.DataFrame(np.stack(graph_dists))
        fdf = pd.concat([fdf, dist], axis=1).to_numpy()
    graph_feat = MinMaxScaler().fit_transform(fdf)
    graph_feat = torch.from_numpy(graph_feat).float()
    return graph_feat

def compute_graph_dist(graphs, topk=8):
    node_dists = []
    graph_dists = []
    for g in tqdm(graphs):
        nfeat = g['node_feat']
        centers = nfeat[:, -3:]
        dist = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2) # (n, n)
        assert dist.shape == (len(centers), len(centers))
        node_dist = np.sort(dist, axis=1)[:, -topk:]
        graph_dist = np.sort(node_dist.flatten())[-topk:]
        node_dists.append(node_dist)
        graph_dists.append(graph_dist)
    return node_dists, graph_dists

def load_data(args, data_args, logger, model_mode='inner'):
    pkl_path, df_path, feat_path, dist_path = load_datapath(data_args.file_dir_postfix, model_mode)
    logger.info(f'Loading data from {pkl_path} and {df_path}.')
    graphs = pickle.load(open(pkl_path, "rb"))
    df = pd.read_csv(df_path)
    feat_df = pd.read_csv(feat_path)
    if not os.path.exists(dist_path):
    # if True:
        logger.info('Compute graph distances and node distances.')
        node_dists, graph_dists = compute_graph_dist(graphs, topk=8)
        with open(dist_path, 'wb') as file:
            pickle.dump([node_dists, graph_dists], file)
    node_dists, graph_dists = pickle.load(open(dist_path, 'rb'))

    gid = [g['file_id'] for g in graphs]
    did = df['file_id'].tolist()
    assert np.all([i0 == i1 for i0, i1 in zip(gid, did)])
    assert df['file_id'].equals(feat_df['file_id'])
    logger.info('We check that dat.csv, graph.pkl, and feature.csv share the same file_id order.')

    logger.info('Firstly, we compute the characteristic crushing strength and the Weibull modulus of a particle, grouped by (diameter, scale_x, scale_y, scale_z, axis_x, axis_y, axis_z).')
    # We add four columns (type, mask, sigma0, modulus), where type refers to the particle type of
    # the 7-tuple, mask indicates whether the particle is useful, sigma0 refers to the 
    # characteristic crushing strength, and modulus refers to the Weibull modulus.
    df = compute_sigma(df, rotate=args.rotate)
    if args.without_gravity:
        logger.info('The particle features remove the gravity.')
        feat_df.pop('gravity')
    logger.info('Set %s as the training task', args.task)
    if args.task == 'sigma0':
        Y = df['sigma0'].to_numpy() / 1e6
    elif args.task == 'modulus':
        Y = df['modulus'].to_numpy()
    else:
        raise NotImplementedError(args.task)
    Y = torch.from_numpy(Y).float()

    logger.info('Loading graph_feat.')    
    fdf = feat_df.drop('file_id', axis=1)
    if args.using_gdist:
        logger.info('Using graph_dist in the GNN model.')
        dist = pd.DataFrame(np.stack(graph_dists))
        fdf = pd.concat([fdf, dist], axis=1).to_numpy()
    graph_feat = MinMaxScaler().fit_transform(fdf)
    graph_feat = torch.from_numpy(graph_feat).float()

    for g in graphs:
        if args.remove_nfeat:
            g['node_feat'] = np.zeros_like(g['node_feat'])
        if args.remove_efeat:
            g['edge_feat'] = np.zeros_like(g['edge_feat'])

    if args.using_ndist:
        logger.info('Using node_dist in the GNN model.')
        for g, d in zip(graphs, node_dists):
            g['node_feat'] = np.concatenate([g['node_feat'], d], axis=1)

    nfeat_scaler = MinMaxScaler()
    efeat_scaler = MinMaxScaler()
    for i, g in enumerate(graphs):
        nfeat_scaler.partial_fit(g['node_feat'])
        efeat_scaler.partial_fit(g['edge_feat'])

    logger.info('Transform node feat and edge feat into PyG graph.')
    pyg_graphs = []
    for i, (g, feat) in enumerate(zip(graphs, graph_feat)):
        nfeat, efeat, eidx = g['node_feat'], g['edge_feat'], g['edge_index']
        nfeat = nfeat_scaler.transform(nfeat)
        efeat = efeat_scaler.transform(efeat)
        nfeat = torch.from_numpy(nfeat).float()
        efeat = torch.from_numpy(efeat).float()
        eidx = torch.from_numpy(eidx).long()

        y = Y[i]

        g_ = Data(x=nfeat, edge_index=eidx, edge_attr=efeat, y=y, file_id=g['file_id'])
        pyg_graphs.append((feat, g_))

    train_mask, valid_mask, test_mask = train_test_split(args, data_args, df, choice=args.test_choice)
    logger.info('Old training: %d, validation: %d, test: %d.', sum(train_mask), sum(valid_mask), sum(test_mask))
    fail_mask = df['mask'].to_numpy()
    logger.info('We discard %d particles due to their missing crushing strength.', sum(~fail_mask))
    train_mask = train_mask & fail_mask
    valid_mask = valid_mask & fail_mask
    test_mask = test_mask & fail_mask
    logger.info('New training: %d, validation: %d, test: %d.', sum(train_mask), sum(valid_mask), sum(test_mask))

    tot_mask = train_mask | valid_mask | test_mask
    logger.info('Dataset: %d, particle types: %d.', sum(tot_mask), len(np.unique(Y[tot_mask])))
    logger.info('The distribution of %s in the used dataset.', args.task)
    print(pd.Series(Y[tot_mask]).describe())
    logger.info('The distribution of %s in the test dataset.', args.test_choice)
    print(pd.Series(Y[test_mask]).describe())

    train_data = [g for m, g in zip(train_mask, pyg_graphs) if m]
    valid_data = [g for m, g in zip(valid_mask, pyg_graphs) if m]
    test_data = [g for m, g in zip(test_mask, pyg_graphs) if m]
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(test_data, batch_size=min(args.batch_size, 64), shuffle=False)

    gfeat_dim, nfeat_dim, efeat_dim = graph_feat.shape[1], nfeat.shape[1], efeat.shape[1]

    return trainloader, validloader, testloader, gfeat_dim, nfeat_dim, efeat_dim
