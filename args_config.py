import argparse
from easydict import EasyDict

def train_args():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument('--gpu-id', type=int, default=0)
    # learning
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--min-epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.00001)
    # task setting
    parser.add_argument('--task', default='sigma0', choices=['sigma0', 'modulus'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--test-choice', default='diameter', choices=['diameter', 'scale', 'rotation'], help='Use which kind to test the model generalization.')
    parser.add_argument('--no-rotate', dest='rotate', action='store_false', help='Whether use rotation dataset.')
    parser.add_argument('--valid-ratio', default=0.2)
    parser.add_argument('--test-ratio', default=0.37)
    # model selection
    parser.add_argument('--model', type=str, default='MLP')
    # use features
    parser.add_argument('--model-mode', default='inner', choices=['inner'])
    parser.add_argument('--without-gravity', action='store_true', help='Whether use the gravity of the particle features.')
    parser.add_argument('--using-gfeat', action='store_true')
    parser.add_argument('--using-gdist', action='store_true')
    parser.add_argument('--using-ndist', action='store_true')
    parser.add_argument('--remove-nfeat', action='store_true')
    parser.add_argument('--remove-efeat', action='store_true')

    # args = parser.parse_args()
    return parser


def particle_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ns',
        '--num-seeds',
        type=int,
        default=20,
        help=
        'We choose to simulate the crushing particle with 20 different diameters.'
    )
    parser.add_argument(
        '-dmin',
        '--diameter-min',
        type=float,
        default=1.0,
        help='We start the diameter from 1.0*(0.016*1.3672*0.542).')
    parser.add_argument(
        '-dmax',
        '--diameter-max',
        type=float,
        default=3.0,
        help='We start the diameter from 3.0*(0.016*1.3672*0.542).')
    parser.add_argument(
        '-nt',
        '--num-tests',
        type=int,
        default=50,
        help=
        'For each kind of particle, we set 50 particles to avoid the failed tests.'
    )
    parser.add_argument('--neper', action='store_true')
    parser.add_argument('--no-rotate', dest='rotate', action='store_false')
    parser.add_argument('--lmgc', action='store_true')
    parser.add_argument('--lmgc-idx', type=int, default=0, help='We choose which scale_choice to compute on the multi-core server.')
    parser.add_argument('--post', action='store_true')

    args = parser.parse_args(args=[])
    args.file_dir_postfix = 'seed{}_dmin{:.0f}_dmax{:.0f}_test{}_rotate{}'.format(
        args.num_seeds, args.diameter_min, args.diameter_max, args.num_tests,
        args.rotate)
    args.init_diameter = 0.016 * 1.3672 * 0.542
    args.scale_choices = [(1, 1, 1), (1/0.95,0.95,1), (1/0.9,0.9,1),\
        (1.1,1.1,1), (1.25,1.21/1.25,1), (1.21/0.9,0.9,1),\
        (1.2,1.2,1), (1.25,1.44/1.25,1), (1.5,1.44/1.5,1),\
        (1.3,1.3,1), (1.25,1.69/1.25,1), (1.5,1.69/1.5,1),\
        (1.4,1.4,1), (1.25,1.96/1.25,1), (1.5,1.96/1.5,1)]
    return args
