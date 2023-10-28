from __future__ import annotations

import argparse
import importlib
import pytorch_lightning as pl

from datetime import datetime
from pathlib import Path
from ray.tune.schedulers import AsyncHyperBandScheduler
from typing import Any

from lightning_ray import LightningTuner, parse_args_dynamic
from loader import get_concept_loss_fn, get_dummy_batch, get_datamodule, DATASET_INFO
from models import *
from ray_utils import RayConfig
from utils import cross_correlation



def make_concept_model(**config) -> ConceptLightningModel:
    """
    Create a concept model.

    Parameters
    ----------
    experiment_module_name : str
        Name of the experiment module (e.g. 'experiments.cifar')
    model_type : str
        Model type
    training_mode : one of {'independent', 'sequential', 'joint'}
        Training mode (see https://arxiv.org/abs/2007.04612)
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
    cw_alignment_frequency : int
        Frequency of concept alignment for whitening (in epochs)
    """
    experiment_module = importlib.import_module(config['experiment_module_name'])
    model_type = config['model_type']

    # Update config with any missing dataset information (e.g. concept_dim, num_classes)
    dataset_info = DATASET_INFO[config['dataset']]
    config = {**dataset_info, **config}

    # Get concept loss function
    config['concept_loss_fn'] = get_concept_loss_fn(
        config['dataset'], config['data_dir'])

    # No residual
    if model_type == 'no_residual':
        config = {**config, 'residual_dim': 0}
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    # With latent residual
    elif model_type in 'latent_residual':
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    # With decorrelated residual
    elif model_type == 'decorrelated_residual':
        residual_loss_fn = lambda r, c: cross_correlation(r, c).square().mean()
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(
            model, residual_loss_fn=residual_loss_fn, **config
        )

    # With MI-minimized residual
    elif model_type == 'mi_residual':
        model = experiment_module.make_concept_model(config)
        model = MutualInfoConceptLightningModel(model, **config)

    # With iterative normalization
    elif model_type == 'iter_norm':
        config = {**config, 'norm_type': 'iter_norm'}
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    # With concept whitening
    elif model_type == 'concept_whitening':
        config = {
            **config,
            'concept_type': 'continuous',
            'norm_type': 'concept_whitening',
            'training_mode': 'joint',
        }
        model = experiment_module.make_concept_model(config)
        model = ConceptWhiteningLightningModel(model, **config)

    else:
        raise ValueError("Unknown model type:", model_type)

    # Dummy pass to handle any un-initialized parameters
    batch = get_dummy_batch(config['dataset'], config['data_dir'])
    model.dummy_pass([batch])

    return model



class ConceptModelTuner(LightningTuner):
    """
    Use Ray Tune to train concept models.
    """

    def get_datamodule(self, config: dict[str, Any]) -> pl.LightningDataModule:
        return get_datamodule(
            dataset_name=config['dataset'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_cpus', 1) - 1,
        )

    def get_model(self, config: dict[str, Any]) -> pl.LightningModule:
        if getattr(self, 'model', None) is None:
            self.model = make_concept_model(**config)

        return self.model

    def get_callbacks(self, config: dict[str, Any]) -> list[pl.Callback]:
        if getattr(self, 'model', None) is None:
            self.model = self.get_model(config)

        return [self.model.callback()]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='experiments.pitfalls',
        help="Experiment configuration module")
    parser.add_argument(
        '--data-dir', type=str, help="Directory where data is stored")
    parser.add_argument(
        '--save-dir', type=str, help="Directory to save models to")
    parser.add_argument(
        '--num-workers', type=int, help="Number of workers to use (per model)")
    parser.add_argument(
        '--num-cpus', type=float, help="Number of CPUs to use (per worker)")
    parser.add_argument(
        '--num-gpus', type=float, help="Number of GPUs to use (per worker)")
    parser.add_argument(
        '--groupby', type=str, nargs='+', help="Config keys to group by")

    args, args_config = parse_args_dynamic(parser)

    # Load experiment config
    experiment_module = importlib.import_module(args.config)
    config = RayConfig(experiment_module.get_config())
    config.set('experiment_module_name', args.config)
    config.set('data_dir', Path(config.get('data_dir')).expanduser().resolve())
    config.set('save_dir', Path(config.get('save_dir')).expanduser().resolve())

    # Override config with any command-line arguments
    config.update({k: v for k, v in vars(args).items() if v is not None})
    config.update(args_config)

    # Download datasets (if necessary) before launching Ray Tune
    # Avoids each initial worker trying to downloading the dataset simultaneously
    dataset_names = config.get('dataset')
    if isinstance(dataset_names, dict) and 'grid_search' in dataset_names:
        dataset_names = list(dataset_names.values())
    dataset_names = [dataset_names] if isinstance(dataset_names, str) else dataset_names
    for dataset_name in dataset_names:
        get_datamodule(dataset_name, data_dir=config.get('data_dir'))

    # Get experiment name
    date = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_name = config.get('experiment_module_name').split('.')[-1]
    experiment_name = f"{experiment_name}/{date}/train"

    # Train models
    config.set('max_epochs', config.get('num_epochs'))
    scheduler = AsyncHyperBandScheduler(max_t=config.get('max_epochs'), grace_period=10)
    tuner = ConceptModelTuner(metric='val_acc', mode='max', scheduler=scheduler)
    tuner.fit(
        param_space=config,
        save_dir=args.save_dir or config.get('save_dir'),
        experiment_name=experiment_name,
        num_workers_per_trial=config.get('num_workers', 1),
        num_cpus_per_worker=config.get('num_cpus', 1),
        num_gpus_per_worker=config.get('num_gpus', 1),
        groupby=args.groupby or config.get('groupby', []),
    )
