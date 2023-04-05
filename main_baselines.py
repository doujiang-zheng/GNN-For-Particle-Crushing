import argparse
import os
import pickle
from random import shuffle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, r2_score
import xgboost as xgb
import lightgbm as lgb

from args_config import particle_args, train_args
from data_util import compute_sigma, compute_graph_dist, load_datapath, train_test_split
from main_nn import test_metrics
from model_config import get_model_config
from util import set_logger, set_random_seed, write_result


def fit_model(name, model_args, train_X, train_y):
    if name == 'Linear':
        model = LinearRegression(**model_args).fit(train_X, train_y)
    elif name == 'Ridge':
        model = Ridge(**model_args).fit(train_X, train_y)
    elif name == 'RF':
        model = RandomForestRegressor(**model_args).fit(train_X, train_y)
    elif name == 'XGB':
        model = xgb.XGBRegressor(**model_args).fit(train_X, train_y)
    elif name == 'LGB':
        model = lgb.LGBMRegressor(**model_args).fit(train_X, train_y)

    return model


def main(args, data_args, model_args):
    logger = set_logger(log_file=True, name=args.model)
    logger.info(args)
    logger.info(data_args)
    logger.info(model_args)
    set_random_seed(args.seed)

    _, df_path, feat_path, dist_path = load_datapath(data_args.file_dir_postfix, args.model_mode)
    logger.info('Loading data from %s.', df_path)
    df = pd.read_csv(df_path)
    feat_df = pd.read_csv(feat_path)
    node_dists, graph_dists = pickle.load(open(dist_path, 'rb'))

    assert np.all(df['file_id'].equals(feat_df['file_id']))
    logger.info('We check that dat.csv and feature.csv share the same file_id order.')
    logger.info('Loading graph_feat.')    
    fdf = feat_df.drop('file_id', axis=1)
    dist = pd.DataFrame(np.stack(graph_dists))
    if args.using_gfeat and args.using_gdist:
        logger.info('Using graph_dist in the ML model.')
        fdf = pd.concat([fdf, dist], axis=1).to_numpy()
    elif not args.using_gfeat and args.using_gdist:
        logger.info('Only use graph_dist in the ML model.')
        fdf = dist.to_numpy()
    elif args.using_gfeat and not args.using_gdist:
        fdf = fdf.to_numpy()
    else:
        fdf = np.zeros_like(fdf)
    X = MinMaxScaler().fit_transform(fdf)
    logger.info('Particle features %d.', X.shape[1])

    # Add four new columns: sigma0, modulus, type, mask
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

    train_mask, valid_mask, test_mask = train_test_split(args, data_args, df, choice=args.test_choice)
    logger.info('Old training: %d, validation: %d, test: %d.', sum(train_mask), sum(valid_mask), sum(test_mask))
    fail_mask = df['mask'].to_numpy()
    logger.info('We discard %d particles due to their missing crushing strength.', sum(~fail_mask))
    train_mask = train_mask & fail_mask
    valid_mask = valid_mask & fail_mask
    test_mask = test_mask & fail_mask
    logger.info('New training: %d, validation: %d, test: %d.', sum(train_mask), sum(valid_mask), sum(test_mask))
   
    tot_mask = train_mask | valid_mask | test_mask
    logger.info('The distribution of %s in the used dataset.', args.task)
    print(pd.Series(Y[tot_mask]).describe())

    train_X, train_y = X[train_mask], Y[train_mask]
    valid_X, valid_y = X[valid_mask], Y[valid_mask]
    test_X, test_y = X[test_mask], Y[test_mask]
    model = fit_model(args.model, model_args, train_X, train_y)
    train_y_hat = model.predict(train_X)
    trm = test_metrics(train_y, train_y_hat)
    valid_y_hat = model.predict(valid_X)
    vam = test_metrics(valid_y, valid_y_hat)
    test_y_hat = model.predict(test_X)
    tem = test_metrics(test_y, test_y_hat)
    logger.info('Train MAE: %.4f, RMSE: %.4f, R2: %.4f', trm['MAE'], trm['RMSE'], trm['R2'])
    logger.info('Valid MAE: %.4f, RMSE: %.4f, R2: %.4f', vam['MAE'], vam['RMSE'], vam['R2'])
    logger.info('Test MAE: %.4f, RMSE: %.4f, R2: %.4f', tem['MAE'], tem['RMSE'], tem['R2'])

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
    assert args.model in ['Linear', 'Ridge', 'RF', 'XGB', 'LGB'], f'Attention: {args.model} not in Linear, Ridge, RF, XGB, LGB.'
    model_args = get_model_config(args.model)[0]

    main(args, data_args, model_args)