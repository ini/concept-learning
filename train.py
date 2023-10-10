from __future__ import annotations

import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from pathlib import Path
from ray import air, tune
from torch.utils.data import DataLoader
from typing import Any, Callable

from lib.club import CLUB
from loader import get_data_loaders, DATASET_NAMES
from models import ConceptModel, ConceptWhiteningModel
from utils import (
    accuracy,
    cross_correlation,
    disable_ray_storage_context,
    get_cw_callback_fn,
    get_mi_callback_fn,
    train_multiclass_classification,
)



### Creating & Training Concept Models

def make_concept_model(
    experiment_module_name: str,
    model_type: str,
    concept_dim: int,
    residual_dim: int,
    mi_estimator_hidden_dim: int,
    mi_optimizer_lr: float,
    whitening_alignment_frequency: int,
    whitening_alignment_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs) -> tuple[ConceptModel, Callable, Callable]:
    """
    Create a concept model.

    Parameters
    ----------
    experiment_module_name : str
        Name of the experiment module (e.g. 'experiments.cifar')
    model_type : str
        Model type
    concept_dim : int
        Size of concept vector
    residual_dim : int
        Size of residual vector
    mi_estimator_hidden_dim : int
        Hidden dimension of mutual information estimator
    mi_optimizer_lr : float
        Learning rate of mutual information estimator optimizer
    whitening_alignment_frequency : int
        Frequency of concept alignment for whitening (in epochs)
    whitening_alignment_loader: DataLoader,
        Data loader to use for concept alignment for whitening
    device : str
        Device to load model on

    Returns
    -------
    model : ConceptModel
        Concept model
    callback_fn : Callable(model, epoch, batch_index, batch)
        Callback function triggered before each batch
    residual_loss_fn : Callable(residual, concept_preds) -> Tensor
        Function to compute the residual loss
    """
    config = locals().copy()
    config.update(kwargs)

    # Load experiment module
    experiment_module = importlib.import_module(experiment_module_name)
    make_bottleneck_model_fn = experiment_module.make_bottleneck_model
    make_whitening_model_fn = experiment_module.make_whitening_model

    # Initialize callback function and residual loss function
    callback_fn = lambda *args, **kwargs: None
    residual_loss_fn = lambda r, c: torch.tensor(0)

    # No residual
    if model_type == 'no_residual':
        model = make_bottleneck_model_fn(dict(config, residual_dim=0)).to(device)

    # With latent residual
    elif model_type == 'latent_residual':
        model = make_bottleneck_model_fn(config).to(device)

    # With decorrelated residual
    elif model_type == 'decorrelated_residual':
        model = make_bottleneck_model_fn(config).to(device)
        residual_loss_fn = lambda r, c: cross_correlation(r, c).square().mean()

    # With MI-minimized residual
    elif model_type == 'mi_residual':
        model = make_bottleneck_model_fn(config).to(device)
        mi_estimator = CLUB(
            residual_dim,
            concept_dim,
            mi_estimator_hidden_dim,
        ).to(device)
        mi_optimizer = optim.Adam(
            mi_estimator.parameters(), lr=mi_optimizer_lr)
        callback_fn = get_mi_callback_fn(mi_estimator, mi_optimizer)
        residual_loss_fn = mi_estimator.forward

    elif model_type == 'whitened_residual':
        model = make_whitening_model_fn(config).to(device)
        callback_fn=get_cw_callback_fn(
            whitening_alignment_loader, concept_dim,
            alignment_frequency=whitening_alignment_frequency,
        )

    else:
        raise ValueError('Unknown model type:', model_type)

    return model, callback_fn, residual_loss_fn

def train_concept_model(
    model: ConceptModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    alpha: float,
    beta: float,
    callback_fn: Callable,
    residual_loss_fn: Callable,
    checkpoint_frequency: int,
    verbose: bool = False,
    **kwargs):
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
    num_epochs : int
        Number of epochs to train for
    lr : float
        Learning rate
    alpha : float
        Weight of concept loss
    beta : float
        Weight of residual loss
    callback_fn : Callable(model, epoch, batch_index, batch)
        Callback function triggered before each batch
    residual_loss_fn : Callable(residual, concept_preds) -> Tensor
        Function to compute the residual loss
    checkpoint_frequency : int
        How often to create model checkpoints (in epochs)
    verbose : bool
        Whether to print training progress
    """
    # Define loss function
    if isinstance(model, ConceptWhiteningModel):
        loss_fn = lambda batch, outputs: nn.CrossEntropyLoss()(outputs[-1], batch[1])
    else:
        def loss_fn(batch, outputs):
            (_, concept_targets), targets = batch
            concept_preds, residual, target_logits = outputs
            concept_targets = concept_targets[..., :concept_preds.shape[-1]]
            concept_loss = nn.BCELoss()(concept_preds, concept_targets)
            residual_loss = residual_loss_fn(residual, concept_preds)
            target_loss = nn.CrossEntropyLoss()(target_logits, targets)
            return (alpha * concept_loss) + (beta * residual_loss) + target_loss

    # Train the model
    train_multiclass_classification(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        callback_fn=callback_fn,
        loss_fn=loss_fn,
        predict_fn=lambda outputs: outputs[-1].argmax(dim=-1),
        save_path='./model.pt',
        checkpoint_frequency=checkpoint_frequency,
        verbose=verbose,
    )

    # Test the model
    acc = accuracy(
        model, val_loader,
        predict_fn=lambda outputs: outputs[-1].argmax(dim=-1),
    )
    if verbose:
        print('Validation Classification Accuracy:', acc)



### Ray Training Configuration

def config_get(config: dict[str, Any], key: str):
    """
    Get a value from a Ray-style configuration dictionary
    (handles top-level grid search).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    key : str
        Configuration key
    """
    if 'grid_search' in config:
        return config['grid_search'][0][key]
    else:
        return config[key]

def config_set(config: dict[str, Any], key: str, value: Any):
    """
    Set a value in a Ray-style configuration dictionary
    (handles top-level grid search).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    key : str
        Configuration key
    value : Any
        Configuration value
    """
    if 'grid_search' in config:
        for config in config['grid_search']:
            config[key] = value
    else:
        config[key] = value

def get_train_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Get Ray-style train configuration dictionary from command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Load provided experiment config
    experiment_module = importlib.import_module(args.config)
    config = experiment_module.get_config()
    config_set(config, 'experiment_module_name', args.config)

    # Override config with command line arguments
    for key, value in vars(args).items():
        if isinstance(value, list):
            config_set(config, key, tune.grid_search(value))
        elif value is not None:
            config_set(config, key, value)

    # Use absolute paths
    config_set(config, 'data_dir', Path(config_get(config, 'data_dir')).resolve())
    config_set(config, 'save_dir', Path(config_get(config, 'save_dir')).resolve())

    return config

def train(config: dict[str, Any]):
    """
    Create and train a model with the given configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
        (see arguments to `make_concept_model()` and `train_concept_model()`)
    """
    # Get data loaders
    train_loader, val_loader, _, concept_dim, num_classes = get_data_loaders(
        config['dataset'], data_dir=config['data_dir'], batch_size=config['batch_size'])

    # Update config with dataset information
    config['concept_dim'] = concept_dim
    config['num_classes'] = num_classes

    # Create model, callback function, and residual loss function
    model, callback_fn, residual_loss_fn = make_concept_model(
        whitening_alignment_loader=train_loader, **config)

    # Train the model
    train_concept_model(
        model, train_loader, val_loader,
        callback_fn=callback_fn,
        residual_loss_fn=residual_loss_fn,
        **config,
    )



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
    config = get_train_config(args)

    # Get experiment name
    date = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_name = config_get(config, 'experiment_module_name').split('.')[-1]
    experiment_name = f'{experiment_name}/{date}/train'

    # Get number of GPUs
    try:
        num_gpus = config_get(config, 'num_gpus')
    except:
        num_gpus = 1 if torch.cuda.is_available() else 0

    # Train the model(s)
    tuner = tune.Tuner(
        tune.with_resources(train, resources={'cpu': 1, 'gpu': num_gpus}),
        param_space=config,
        tune_config=tune.TuneConfig(metric='val_acc', mode='max', num_samples=1),
        run_config=air.RunConfig(
            name=experiment_name,
            storage_path=config['save_dir'],
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute='val_acc',
                checkpoint_score_order='max',
                num_to_keep=5,
            ),
        ),
    )
    results = tuner.fit()
