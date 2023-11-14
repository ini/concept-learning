from __future__ import annotations

import argparse
import importlib
import pytorch_lightning as pl

from datetime import datetime
from pathlib import Path
from ray.tune.schedulers import AsyncHyperBandScheduler

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
    model_type = config.get('model_type', 'latent_residual')

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
    elif model_type == 'latent_residual':
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    # With decorrelated residual
    elif model_type == 'decorrelated_residual':
        residual_loss_fn = lambda r, c: cross_correlation(r, c).square().mean()
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, residual_loss_fn=residual_loss_fn, **config)

    # With MI-minimized residual
    elif model_type == 'mi_residual':
        model = experiment_module.make_concept_model(config)
        model = MutualInfoConceptLightningModel(model, **config)

    # With iterative / layer normalization
    elif model_type in ('iter_norm', 'layer_norm'):
        config = {**config, 'norm_type': model_type}
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

def make_datamodule(**config) -> pl.LightningDataModule:
    return get_datamodule(
        dataset_name=config['dataset'],
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=int(config.get('num_cpus', 1)) - 1,
        resize_oai=config.get('resize_oai', True),
    )



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
        '--restore-path', type=str, help="Path to restore model from")
    parser.add_argument(
        '--groupby', type=str, nargs='+', help="Config keys to group by")
    parser.add_argument(
        '--scheduler', action='store_true', help="Use Tune trial scheduler")

    args, args_config = parse_args_dynamic(parser)

    # Load experiment config
    experiment_module = importlib.import_module(args.config)
    config = RayConfig(experiment_module.get_config())
    config.update({k: v for k, v in vars(args).items() if v is not None})
    config.update(args_config)
    config.set('experiment_module_name', args.config)
    config.set('data_dir', Path(config.get('data_dir')).expanduser().resolve())
    config.set('save_dir', Path(config.get('save_dir')).expanduser().resolve())

    # Download datasets (if necessary) before launching Ray Tune
    # Avoids each initial worker trying to download the dataset simultaneously
    dataset_names = config.get('dataset')
    if isinstance(dataset_names, dict) and 'grid_search' in dataset_names:
        dataset_names = list(dataset_names.values())
    dataset_names = [dataset_names] if isinstance(dataset_names, str) else dataset_names
    for dataset_name in dataset_names:
        get_datamodule(dataset_name, data_dir=config.get('data_dir'))

    # Create trial scheduler
    scheduler = None
    if args.scheduler:
        scheduler = AsyncHyperBandScheduler(
            max_t=config.get('num_epochs'), grace_period=config.get('num_epochs') // 5)

    # Get experiment name
    date = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_name = config.get('experiment_module_name').split('.')[-1]
    experiment_name = f"{experiment_name}/{date}/train"

    # Train models
    config.set('max_epochs', config.get('num_epochs'))
    if args.restore_path:
        tuner = LightningTuner.restore(args.restore_path, resume_errored=True)
    else:
        tuner = LightningTuner(
            metric='val_acc',
            mode='max',
            scheduler=scheduler,
            num_samples=config.get('num_samples', 1),
        )
    tuner.fit(
        make_concept_model,
        make_datamodule,
        param_space=config,
        save_dir=args.save_dir or config.get('save_dir'),
        experiment_name=experiment_name,
        num_workers_per_trial=config.get('num_workers', 1),
        num_cpus_per_worker=config.get('num_cpus', 1),
        num_gpus_per_worker=config.get('num_gpus', 1),
        gpu_memory_per_worker=config.get('gpu_memory_per_worker', None),
        groupby=config.get('groupby', []),
    )
