import argparse
import importlib
import os
import pytorch_lightning as pl
import torch

from datetime import datetime
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray import tune
from ray.air import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy, RayFSDPStrategy, RayDeepSpeedStrategy,
    RayLightningEnvironment,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune import TuneConfig, Tuner
from typing import Any

from loader import get_data_loaders, DATASET_NAMES
from models import *
from ray_utils import config_get, config_set, RayCallback
from utils import cross_correlation



def get_train_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Get Ray experiment configuration dictionary from command line arguments.

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
    config_set(
        config, 'data_dir', Path(config_get(config, 'data_dir')).expanduser().resolve())
    config_set(
        config, 'save_dir', Path(config_get(config, 'save_dir')).expanduser().resolve())

    return config

def make_concept_model(**config) -> ConceptLightningModel:
    """
    Create a concept model.

    Parameters
    ----------
    experiment_module_name : str
        Name of the experiment module (e.g. 'experiments.cifar')
    model_type : str
        Model type
    device : str
        Device to load model on
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
    """
    experiment_module_name = config.get('experiment_module_name')
    model_type = config.get('model_type', 'no_residual')
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load experiment module
    experiment_module = importlib.import_module(experiment_module_name)
    make_bottleneck_model_fn = experiment_module.make_bottleneck_model

    # No residual
    if model_type == 'no_residual':
        model = make_bottleneck_model_fn(dict(config, residual_dim=0)).to(device)
        model = ConceptLightningModel(model, **config)

    # With latent residual
    elif model_type in 'latent_residual':
        model = make_bottleneck_model_fn(config).to(device)
        model = ConceptLightningModel(model, **config)

    # With decorrelated residual
    elif model_type == 'decorrelated_residual':
        residual_loss_fn = lambda r, c: cross_correlation(r, c).square().mean()
        model = make_bottleneck_model_fn(config).to(device)
        model = ConceptLightningModel(model, residual_loss_fn=residual_loss_fn, **config)

    # With MI-minimized residual
    elif model_type == 'mi_residual':
        model = make_bottleneck_model_fn(config).to(device)
        model = MutualInfoConceptLightningModel(model, **config)

    # With concept whitening
    elif model_type == 'concept_whitening':
        make_whitening_model_fn = experiment_module.make_whitening_model
        model = make_whitening_model_fn(config).to(device)
        model = ConceptWhiteningLightningModel(model, **config)

    else:
        raise ValueError('Unknown model type:', model_type)

    return model

def train(config: dict[str, Any]):
    """
    Train a concept model.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    """
    # Get data loaders
    train_loader, val_loader, _, concept_dim, num_classes = get_data_loaders(
        config['dataset'], data_dir=config['data_dir'], batch_size=config['batch_size'])

    # Update config with dataset information
    config['concept_dim'] = concept_dim
    config['num_classes'] = num_classes

    # Create model
    model = make_concept_model(**config)
    model.dummy_pass(train_loader)

    # Train model
    trainer = pl.Trainer(
        accelerator='cpu' if MPSAccelerator.is_available() else 'auto',
        strategy=RayDDPStrategy(),
        devices='auto',
        logger=False, # logging metrics is handled by Ray
        callbacks=[model.callback(), RayCallback(**config)],
        max_epochs=config['num_epochs'],
        enable_checkpointing=False, # checkpointing is handled by Ray
        enable_progress_bar=False,
        plugins=[RayLightningEnvironment()],
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_loader, val_loader)



if __name__ == '__main__':
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
            'concept_whitening',
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

    # Set Ray storage directory
    os.environ.setdefault('RAY_AIR_LOCAL_CACHE_DIR', str(config['save_dir']))

    # Create Ray trainer
    ray_trainer = TorchTrainer(
        train,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=(num_gpus > 0),
            resources_per_worker={'CPU': 1, 'GPU': num_gpus},
        ),
    )

    # Train the model(s)
    tuner = Tuner(
        ray_trainer,
        param_space={'train_loop_config': config},
        tune_config=TuneConfig(metric='val_acc', mode='max', num_samples=1),
        run_config = RunConfig(
            name=experiment_name,
            storage_path=config['save_dir'],
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute='val_acc',
                checkpoint_score_order='max',
            ),
        )
    )
    results = tuner.fit()
