from __future__ import annotations

import argparse
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import ray.train
import ray.train.context

from copy import deepcopy
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray import air, tune
from torch import Tensor
from torch.utils.data import DataLoader

from loader import get_datamodule, DATASET_INFO
from nn_extensions import Chain
from models import ConceptLightningModel
from train import ConceptModelTuner
from utils import logit_fn, set_cuda_visible_devices



### Interventions

class Randomize(nn.Module):
    """
    Shuffle data along the batch dimension.
    """

    def forward(self, x: Tensor, *args, **kwargs):
        return x[torch.randperm(len(x))]


class Intervention(nn.Module):
    """
    Intervene on a random subset of concepts.
    """

    def __init__(self, num_interventions: int, negative: bool = False):
        """
        Parameters
        ----------
        num_interventions : int
            Number of concepts to intervene on
        negative : bool
            Whether to intervene with incorrect concept values
        """
        super().__init__()
        self.num_interventions = num_interventions
        self.negative = negative

    def forward(self, x: Tensor, concepts: Tensor):
        if self.negative:
            concepts = 1 - concepts   # flip binary concepts to opposite values

        concept_dim = concepts.shape[-1]
        concept_logits = logit_fn(concepts)
        idx = torch.randperm(concept_dim)[:self.num_interventions]
        x[:, idx] = concept_logits[:, idx]
        return x



### Evaluations

def test(model: pl.LightningModule, loader: DataLoader):
    """
    Test model.

    Parameters
    ----------
    model : pl.LightningModule
        Model to test
    loader : DataLoader
        Test data loader
    """
    trainer = pl.Trainer(
        accelerator='cpu' if MPSAccelerator.is_available() else 'auto',
        enable_progress_bar=False,
    )
    return trainer.test(model, loader)[0]

def test_interventions(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    concept_dim: int,
    negative: bool,
    max_samples = 10) -> float:
    """
    Test model accuracy with concept interventions.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Dimension of concept vector
    negative : bool
        Whether to intervene with incorrect concept values
    max_samples : int
        Maximum number of interventions to test (varying the # of concepts intervened on)
    """
    x = np.linspace(0, concept_dim + 1, num=min(concept_dim + 2, max_samples), dtype=int)
    y = np.zeros(len(x))
    for i, num_interventions in enumerate(x):
        intervention = Intervention(num_interventions, negative=negative)
        new_model = deepcopy(model)
        new_model.concept_model.bottleneck_layer = Chain(
            new_model.concept_model.bottleneck_layer, intervention)
        results = test(new_model, test_loader)
        y[i] = results['test_acc']

    return {'x': x, 'y': y}

def test_random_concepts(model: ConceptLightningModel, test_loader: DataLoader) -> float:
    """
    Test model accuracy with randomized concept predictions.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    # Shuffle data
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=True,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory,
    )

    new_model = deepcopy(model)
    new_model.concept_model.concept_network = Chain(
        new_model.concept_model.concept_network, Randomize())
    results = test(new_model, test_loader)
    return results['test_acc']

def test_random_residual(model: ConceptLightningModel, test_loader: DataLoader) -> float:
    """
    Test model accuracy with randomized residual values.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    # Shuffle data
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=True,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory,
    )

    new_model = deepcopy(model)
    new_model.concept_model.residual_network = Chain(
        new_model.concept_model.residual_network, Randomize())
    results = test(new_model, test_loader)
    return results['test_acc']



### Loading & Execution

def filter_eval_configs(configs: list[dict]) -> list[dict]:
    """
    Filter evaluation configs.

    Parameters
    ----------
    configs : list[dict]
        List of evaluation configs
    """
    configs_to_keep = []
    for config in configs:
        # TODO: support interventions for concept whitening models
        if config['eval_mode'].endswith('intervention'):
            if config['model_type'] == 'concept_whitening':
                print('Interventions not supported for concept whitening models')
                continue

        configs_to_keep.append(config)

    return configs_to_keep

def evaluate(config: dict):
    """
    Evaluate a trained model.

    Parameters
    ----------
    config : dict
        Evaluation configuration dictionary
    """
    metrics = {}

    # Get data loader
    test_loader = get_datamodule(
        config['dataset'],
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=0,
    ).test_dataloader()

    # Load model
    model = ConceptModelTuner('val_acc', 'max').load_model(config['train_result'])

    # Evaluate model
    if config['eval_mode'] == 'accuracy':
        results = test(model, test_loader)
        for key in ('test_acc', 'test_concept_acc'):
            if key in results:
                metrics[key] = results[key]

    if config['eval_mode'] == 'neg_intervention':
        concept_dim = DATASET_INFO[config['dataset']]['concept_dim']
        metrics['neg_intervention_accs'] = test_interventions(
            model, test_loader, concept_dim, negative=True)

    elif config['eval_mode'] == 'pos_intervention':
        concept_dim = DATASET_INFO[config['dataset']]['concept_dim']
        metrics['pos_intervention_accs'] = test_interventions(
            model, test_loader, concept_dim, negative=False)

    elif config['eval_mode'] == 'random_concepts':
        metrics['random_concept_acc'] = test_random_concepts(model, test_loader)

    elif config['eval_mode'] == 'random_residual':
        metrics['random_residual_acc'] = test_random_residual(model, test_loader)

    # Report evaluation metrics
    ray.train.report(metrics)



if __name__ == '__main__':
    MODES = [
        'accuracy',
        'neg_intervention',
        'pos_intervention',
        'random_concepts',
        'random_residual',
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-dir', type=str, default=os.environ.get('CONCEPT_SAVE_DIR', './saved'),
        help='Experiment directory')
    parser.add_argument(
        '--mode', nargs='+', default=MODES, help='Evaluation modes')
    parser.add_argument(
        '--groupby', nargs='+', default=['dataset', 'model_type'],
        help='Config keys to group by when selecting best trial results')
    parser.add_argument(
        '--all', action='store_true',
        help='Evaluate all trained models (instead of best trial per group)')
    parser.add_argument(
        '--num-cpus', type=float, default=1, help='Number of CPUs to use (per model)')
    parser.add_argument(
        '--num-gpus', type=float, default=1, help='Number of GPUs to use (per model)')

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob('**/train/tuner.pkl')
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load train results
    print('Loading training results from', experiment_path / 'train')
    tuner = ConceptModelTuner.restore(experiment_path / 'train')
    if args.all:
        results = tuner.get_results()
    else:
        results = [
            group.get_best_result()
            for group in tuner.get_results(groupby=args.groupby).values()
        ]

    # Create evaluation configs
    eval_configs = filter_eval_configs([
        {
            **result.config['train_loop_config'],
            'train_result': result,
            'eval_mode': mode,
        }
        for result in results
        for mode in args.mode
    ])

    # Get available resources
    if args.num_gpus < 1:
        set_cuda_visible_devices(available_memory_threshold=args.num_gpus)

    # Run evaluations
    tuner = tune.Tuner(
        tune.with_resources(
            evaluate,
            resources={
                'cpu': args.num_cpus,
                'gpu': args.num_gpus if torch.cuda.is_available() else 0
            },
        ),
        param_space=tune.grid_search(eval_configs),
        run_config=air.RunConfig(name='eval', storage_path=experiment_path),
    )
    eval_results = tuner.fit()
