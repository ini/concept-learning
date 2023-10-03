import argparse
import ray.tune as tune
import yaml
import copy
import importlib
import utils
import sys

def create_arg_dict(experiment_config):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'intervention', 'random'],
        help='Mode to run',
    )
    parser.add_argument(
        '--save-dir', type=str, help='Directory to save models to')
    parser.add_argument(
        '--load-dir', type=str, help='Directory to load saved models from')
    parser.add_argument(
        '--num-epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--alpha', type=float, default=1.0, help='Weight for concept loss')
    parser.add_argument(
        '--beta', type=float, default=1.0, help='Weight for residual loss')
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict = utils.merge_dicts(args_dict, experiment_config)
    return args_dict