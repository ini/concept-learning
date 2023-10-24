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
from ray import air, tune
from ray.train import Result
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Iterable

from loader import get_data_loaders, DATASET_INFO
from nn_extensions import Chain
from models import ConceptLightningModel
from ray_utils import group_results
from train import make_concept_model, get_ray_trainer
from utils import logit_fn



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
    y = np.zeros_like(x)
    for i, num_interventions in enumerate(x):
        intervention = Intervention(num_interventions, negative=negative)
        new_model = deepcopy(model)
        new_model.concept_model.bottleneck_layer = Chain(
            new_model.concept_model.bottleneck_layer, intervention)
        trainer = pl.Trainer(enable_progress_bar=False)
        trainer.test(new_model, test_loader)
        y[i] = trainer.callback_metrics['test_acc'].item()

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
    new_model = deepcopy(model)
    new_model.concept_model.concept_network = Chain(
        new_model.concept_model.concept_network, Randomize())
    trainer = pl.Trainer(enable_progress_bar=False)
    trainer.test(new_model, test_loader)
    return trainer.callback_metrics['test_acc'].item()

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
    new_model = deepcopy(model)
    new_model.concept_model.residual_network = Chain(
        new_model.concept_model.residual_network, Randomize())
    trainer = pl.Trainer(enable_progress_bar=False)
    trainer.test(new_model, test_loader)
    return trainer.callback_metrics['test_acc'].item()



### Loading & Execution

def load_train_results(
    path: Path | str,
    best_only: bool = False,
    groupby: list[str] = ['dataset', 'model_type']) -> Iterable[Result]:
    """
    Load train results for the given experiment.

    Parameters
    ----------
    path : Path or str
        Path to the experiment directory
    best_only : bool
        Whether to return only the best result (for each group)
    groupby : list[str]
        List of config keys to group by when selecting best results
    """
    results_path = Path(path).resolve() / 'train'

    # Load all results
    print('Loading training results from', results_path)
    tuner = tune.Tuner.restore(str(results_path), trainable=get_ray_trainer())
    results = tuner.get_results()

    if best_only:
        return [
            group.get_best_result()
            for group in group_results(results, groupby).values()
        ]
    else:
        return results

def load_model(
    result: Result, loader: DataLoader | None = None) -> ConceptLightningModel:
    """
    Load a trained model from a Ray Tune result.

    Parameters
    ----------
    result : Result
        Ray result instance
    loader : DataLoader
        Data loader for dummy pass
    """
    checkpoint_dir = Path(result.get_best_checkpoint('val_acc', 'max').path)
    model = make_concept_model(**result.config['train_loop_config'], loader=loader)
    model.concept_model.load_state_dict(
        torch.load(checkpoint_dir / 'model.pt', map_location=model.device))
    return model

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

    # Get data loaders
    _, _, test_loader = get_data_loaders(
        config['dataset'], data_dir=config['data_dir'], batch_size=config['batch_size'])

    # Load model
    model = load_model(config['train_result'], loader=test_loader)

    # Evaluate model
    if config['eval_mode'] == 'accuracy':
        trainer = pl.Trainer(enable_progress_bar=False)
        trainer.test(model, test_loader)
        metrics['test_acc'] = trainer.callback_metrics['test_acc'].item()
        metrics['test_concept_acc'] = trainer.callback_metrics['test_concept_acc'].item()

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
    results = load_train_results(
        experiment_path, best_only=(not args.all), groupby=args.groupby)

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
