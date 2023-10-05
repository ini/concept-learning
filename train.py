from __future__ import annotations

import argparse
import importlib
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from pathlib import Path
from ray import air, tune
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Any, Callable

from lib.club import CLUB
from loader import get_data_loaders, DATASET_NAMES
from models import ConceptModel, ConceptBottleneckModel, ConceptWhiteningModel
from utils import (
    accuracy,
    cross_correlation,
    disable_ray_storage_context,
    get_cw_callback_fn,
    get_mi_callback_fn,
    train_multiclass_classification,
)



def chain_fns(*fns: Callable) -> Callable:
    """
    Return a chained function that sequentially calls
    the provided functions in the given order.

    Parameters
    ----------
    *fns : Callable
        Functions to chain together
    """
    def chained_fn(*args, **kwargs):
        for fn in fns:
            fn(*args, **kwargs)

    return chained_fn

def concept_model_predict_fn(outputs: Tensor | tuple[Tensor]) -> Tensor:
    """
    Prediction function for concept models.

    Parameters
    ----------
    outputs : Tensor or tuple[Tensor]
        Model outputs
    """
    outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
    return outputs.argmax(dim=-1)

def train_concept_model(
    model: ConceptModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    callback_fn: Callable = lambda model, epoch, batch_index, batch: None,
    residual_loss_fn: Callable = lambda r, c: torch.tensor(0)):
    """
    Train a concept model.

    Parameters
    ----------
    model : ConceptModel
        Model to train
    train_loader : DataLoader
        Train data loader
    val_loader : DataLoader
        Validation data loader
    config : dict[str, Any]
        Configuration dictionary
    callback_fn : Callable(model, epoch, batch_index, batch)
        Callback function triggered before each batch
    residual_loss_fn : Callable(residual, concept_preds) -> Tensor
        Function to compute the residual loss
    """
    # Get callback function
    if isinstance(model, ConceptWhiteningModel):
        cw_callback_fn = get_cw_callback_fn(
            train_loader, config['concept_dim'],
            alignment_frequency=config['whitening_alignment_frequency'],
        )
        callback_fn = chain_fns(callback_fn, cw_callback_fn)

    # Define loss function
    if isinstance(model, ConceptBottleneckModel):
        alpha, beta = config['alpha'], config['beta']
        def loss_fn(batch, outputs):
            (_, concept_targets), targets = batch
            concept_logits, residual, target_logits = outputs
            concept_targets = concept_targets[..., :concept_logits.shape[-1]]
            concept_loss = nn.BCELoss()(concept_logits, concept_targets)
            residual_loss = residual_loss_fn(residual, concept_logits)
            target_loss = nn.CrossEntropyLoss()(target_logits, targets)
            return (alpha * concept_loss) + (beta * residual_loss) + target_loss
    else:
        loss_fn = lambda batch, outputs: nn.CrossEntropyLoss()(outputs, batch[1])

    # Train the model
    train_multiclass_classification(
        model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        callback_fn=callback_fn,
        loss_fn=loss_fn,
        predict_fn=concept_model_predict_fn,
        save_path='./model.pt',
        checkpoint_frequency=config['checkpoint_frequency'],
        verbose=config.get('verbose', False),
    )

    # Test the model
    acc = accuracy(
        model, val_loader,
        predict_fn=concept_model_predict_fn,
    )
    if config.get('verbose', False):
        print('Validation Classification Accuracy:', acc)

def train(config: dict):
    """
    Create and train a model with the given configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    experiment_module = importlib.import_module(config['experiment_module_name'])
    make_bottleneck_model_fn = experiment_module.make_bottleneck_model
    make_whitening_model_fn = experiment_module.make_whitening_model
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    train_loader, val_loader, _, concept_dim, num_classes = get_data_loaders(
        config['dataset'], data_dir=config['data_dir'], batch_size=config['batch_size'])

    # Update config with dataset information
    config['concept_dim'] = concept_dim
    config['num_classes'] = num_classes

    # No residual
    if config['model_type'] == 'no_residual':
        model = make_bottleneck_model_fn(dict(config, residual_dim=0)).to(device)
        train_concept_model(model, train_loader, val_loader, config)

    # With latent residual
    elif config['model_type'] == 'latent_residual':
        model = make_bottleneck_model_fn(config).to(device)
        train_concept_model(model, train_loader, val_loader, config)

    # With decorrelated residual
    elif config['model_type'] == 'decorrelated_residual':
        model = make_bottleneck_model_fn(config).to(device)
        train_concept_model(
            model, train_loader, val_loader, config,
            residual_loss_fn=lambda r, c: cross_correlation(r, c).square().mean(),
        )

    # With MI-minimized residual
    elif config['model_type'] == 'mi_residual':
        model = make_bottleneck_model_fn(config).to(device)
        mi_estimator = CLUB(
            config['residual_dim'],
            config['concept_dim'],
            config['mi_estimator_hidden_dim']
        ).to(device)
        mi_optimizer = optim.Adam(
            mi_estimator.parameters(), lr=config['mi_optimizer_lr'])
        train_concept_model(
            model, train_loader, val_loader, config,
            residual_loss_fn=mi_estimator.forward,
            callback_fn=get_mi_callback_fn(mi_estimator, mi_optimizer),
        )

    # With concept-whitened residual
    elif config['model_type'] == 'whitened_residual':
        model = make_whitening_model_fn(config).to(device)
        train_concept_model(model, train_loader, val_loader, config)

    else:
        raise ValueError('Unknown model type:', config['model_type'])



if __name__ == '__main__':
    disable_ray_storage_context()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='experiments.pitfalls',
        help='Experiment configuration module')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on')
    parser.add_argument(
        '--num-gpus', type=float, help='Number of GPUs to use (per model)')
    parser.add_argument(
        '--groupby', type=str, nargs='+', help='Configuration key to group results by')
    parser.add_argument(
        '--data-dir', type=str, help='Directory where data is stored')
    parser.add_argument(
        '--save-dir', type=str, help='Directory to save models to')
    parser.add_argument(
        '--dataset', type=str, choices=DATASET_NAMES, help='Dataset to train on')
    parser.add_argument(
        '--model-type', type=str, nargs='+',
        choices=[
            'no_residual',
            'latent_residual',
            'decorrelated_residual',
            'mi_residual',
            'whitened_residual',
        ],
        help='Model type',
    )
    parser.add_argument(
        '--residual-dim', type=int, nargs='+', help='Dimensionality of the residual')
    parser.add_argument(
        '--num-epochs', type=int, nargs='+', help='Number of epochs to train for')
    parser.add_argument(
        '--lr', type=float, nargs='+', help='Learning rate')
    parser.add_argument(
        '--batch-size', type=int, nargs='+', help='Batch size')
    parser.add_argument(
        '--alpha', type=float, nargs='+', help='Weight of concept loss')
    parser.add_argument(
        '--beta', type=float, nargs='+', help='Weight of residual loss')
    parser.add_argument(
        '--mi-estimator-hidden-dim', type=int, nargs='+',
        help='Hidden dimension of the MI estimator',
    )
    parser.add_argument(
        '--mi-optimizer-lr', type=float, nargs='+',
        help='Learning rate of the MI estimator optimizer')
    parser.add_argument(
        '--whitening-alignment-frequency', type=int, nargs='+',
        help='Frequency of whitening alignment')
    parser.add_argument(
        '--checkpoint-frequency', type=int, nargs='+', help='Frequency of checkpointing')

    args = parser.parse_args()

    # Load provided experiment config
    experiment_module = importlib.import_module(args.config)
    experiment_config = experiment_module.get_config()
    experiment_config['experiment_module_name'] = args.config

    # Override experiment config with command line arguments
    for key, value in vars(args).items():
        if key == 'groupby':
            continue
        elif isinstance(value, list):
            experiment_config[key] = tune.grid_search(value)
        elif value is not None:
            experiment_config[key] = value

    # Use absolute paths
    experiment_config['data_dir'] = Path(experiment_config['data_dir']).resolve()
    experiment_config['save_dir'] = Path(experiment_config['save_dir']).resolve()

    # Get experiment name
    date = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_name = experiment_module.__name__.split('.')[-1]
    experiment_name = f'{experiment_name}/{date}/train'

    # Create experiment groups by config keys in `args.groupby`,
    # where for each group in `groups`, `group[i]` is the value for the i-th config key.
    # This will run a separate Ray Tune experiment for each group.
    is_grid_search = lambda x: isinstance(x, dict) and 'grid_search' in x and len(x) == 1
    list_values = lambda x: x['grid_search'] if is_grid_search(x) else [x]
    if args.groupby is None:
        groups = [None]
    else:
        groups = list(itertools.product(*(
            list_values(experiment_config[key]) for key in args.groupby)))

    # Train the model(s)
    for group in groups:
        if group is not None:
            for i, key in enumerate(args.groupby):
                experiment_config[key] = group[i]

        group_name = '/'.join(group) if group else None
        num_gpus = experiment_config.get('num_gpus', 1)
        tuner = tune.Tuner(
            tune.with_resources(
                train,
                resources={
                    'cpu': 1,
                    'gpu': num_gpus if torch.cuda.is_available() else 0
                },
            ),
            param_space=experiment_config,
            tune_config=tune.TuneConfig(metric='val_acc', mode='max', num_samples=1),
            run_config=air.RunConfig(
                name=f'{experiment_name}/{group_name}' if group else experiment_name,
                storage_path=experiment_config['save_dir'],
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_score_attribute='val_acc',
                    checkpoint_score_order='max',
                    num_to_keep=5,
                ),
            ),
        )
        results = tuner.fit()
