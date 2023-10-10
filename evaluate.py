import argparse
import importlib
import os
import pyarrow.fs
import torch
import torch.nn as nn
import ray.train
import ray.train.context

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from ray import air, tune
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Sequence

from loader import get_data_loaders
from models import ConceptModel, ConceptWhiteningModel
from train import train
from utils import accuracy, disable_ray_storage_context

import nn_extensions # enable '+' operator for chaining modules



### Helper Modules

class Subscriptable(type):
    """
    Subscriptable metaclass.
    """

    def __getitem__(cls, key):
        return cls(key)


class Randomize(nn.Module, metaclass=Subscriptable):
    """
    Shuffle data along the batch dimension.
    """

    def __init__(self, idx: slice | Sequence[int] = slice(None)):
        super().__init__()
        self.idx = idx

    def forward(self, x: Tensor, *args, **kwargs):
        order = torch.randperm(x.shape[0])
        x[..., self.idx] = x[order][..., self.idx]
        return x


class NegativeIntervention(nn.Module):
    """
    Intervene on a random subset of concepts by setting them to
    the opposite of their ground truth value.
    """

    def __init__(self, num_interventions: int, model_type: type[ConceptModel]):
        super().__init__()
        self.num_interventions = num_interventions
        self.model_type = model_type

    def forward(self, x: Tensor, concepts: Tensor):
        incorrect_concepts = 1 - concepts   # flip binary concept values
        if self.model_type == ConceptWhiteningModel:
            incorrect_concepts = 2 * incorrect_concepts - 1  # concept activations

        concept_dim = concepts.shape[-1]
        intervention_idx = torch.randperm(concept_dim)[:self.num_interventions]
        x[:, intervention_idx] = incorrect_concepts[:, intervention_idx]
        return x


class PositiveIntervention(nn.Module):
    """
    Intervene on a random subset of concepts by setting them to
    their ground truth value.
    """

    def __init__(self, num_interventions: int, model_type: type[ConceptModel]):
        super().__init__()
        self.num_interventions = num_interventions
        self.model_type = model_type

    def forward(self, x: Tensor, concepts: Tensor):
        if self.model_type == ConceptWhiteningModel:
            concepts = 2 * concepts - 1  # convert to concept activations

        concept_dim = concepts.shape[-1]
        intervention_idx = torch.randperm(concept_dim)[:self.num_interventions]
        x[:, intervention_idx] = concepts[:, intervention_idx]
        return x



### Evaluations

def test_interventions(
    model: ConceptModel,
    test_loader: DataLoader,
    intervention_cls: type,
    num_interventions: int) -> float:
    """
    Test model accuracy with concept interventions.

    Parameters
    ----------
    model : ConceptModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    intervention_cls : type
        Intervention class
    num_interventions : int
        Number of concepts to intervene on
    """
    new_model = deepcopy(model)
    new_model.bottleneck_layer += intervention_cls(num_interventions, type(model))
    return accuracy(
        new_model, test_loader, predict_fn=lambda outputs: outputs[-1].argmax(dim=-1))

def test_random_concepts(
    model: ConceptModel, test_loader: DataLoader, concept_dim: int) -> float:
    """
    Test model accuracy with randomized concept predictions.

    Parameters
    ----------
    model : ConceptModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Size of concept vector
    """
    new_model = deepcopy(model)
    new_model.bottleneck_layer += Randomize[:concept_dim]
    return accuracy(
        new_model, test_loader, predict_fn=lambda outputs: outputs[-1].argmax(dim=-1))

def test_random_residual(
    model: ConceptModel, test_loader: DataLoader, concept_dim: int) -> float:
    """
    Test model accuracy with randomized residual values.

    Parameters
    ----------
    model : ConceptModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Size of concept vector
    """
    new_model = deepcopy(model)
    new_model.bottleneck_layer += Randomize[concept_dim:]
    return accuracy(
        new_model, test_loader, predict_fn=lambda outputs: outputs[-1].argmax(dim=-1))



### Loading & Execution

def load_train_results(
    path: Path | str,
    best_only: bool = False,
    groupby: list[str] = ['dataset', 'model_type']) -> list[ray.train.Result]:
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
    tuner = tune.Tuner.restore(str(results_path), trainable=train)
    results = tuner.get_results()
    
    if not best_only:
        return list(results)

    # Group results by config keys in `groupby`
    trials = defaultdict(list)
    for trial in results._experiment_analysis.trials:
        trial.storage = ray.train.context.StorageContext(
            trial.path,
            trial.experiment_dir_name,
            trial_dir_name=trial.path,
        )
        group_key = tuple(trial.config[key] for key in groupby)
        trials[group_key].append(trial)

    # Get best result for each group
    return [
        tune.ResultGrid(
            tune.ExperimentAnalysis(
                results._experiment_analysis.experiment_path,
                storage_filesystem=pyarrow.fs.LocalFileSystem(),
                trials=trials[group_key],
                default_metric=results._experiment_analysis.default_metric,
                default_mode=results._experiment_analysis.default_mode,
            )
        ).get_best_result()
        for group_key in trials
    ]

def load_model(result: ray.train.Result) -> ConceptModel:
    """
    Load a trained model from a Ray Tune result.

    Parameters
    ----------
    result : ray.train.Result
        Ray result instance
    """
    experiment_module = importlib.import_module(result.config['experiment_module_name'])
    make_bottleneck_model = experiment_module.make_bottleneck_model
    make_whitening_model = experiment_module.make_whitening_model

    # Create model
    if result.config['model_type'] == 'no_residual':
        model = make_bottleneck_model(dict(result.config, residual_dim=0))
    elif result.config['model_type'] == 'whitened_residual':
        model = make_whitening_model(result.config)
    else:
        model = make_bottleneck_model(result.config)

    # Load trained model parameters
    model.load_state_dict(torch.load(Path(result.path) / 'model.pt'))
    return model

def evaluate(config: dict):
    """
    Evaluate a trained model.

    Parameters
    ----------
    config : dict
        Evaluation configuration dictionary
    """
    train_result = config['train_result']
    metrics = {}

    # Get data loaders
    _, _, test_loader, concept_dim, _ = get_data_loaders(
        train_result.config['dataset'],
        data_dir=train_result.config['data_dir'],
        batch_size=10000,
    )

    # Load model
    model = load_model(train_result).to(config['device'])

    # Evaluate model
    if config['eval_mode'] == 'accuracy':
        metrics['test_acc'] = accuracy(
            model, test_loader, predict_fn=lambda outputs: outputs[-1].argmax(dim=-1))
        metrics['test_concept_acc'] = accuracy(
            model, test_loader,
            batch_transform_fn=lambda batch: (batch[0][0], batch[0][1]),
            predict_fn=lambda outputs: outputs[0] > 0.5,
        )

    if config['eval_mode'] == 'neg_intervention':
        metrics['neg_intervention_accs'] = [
            test_interventions(model, test_loader, NegativeIntervention, n)
            for n in range(train_result.config['concept_dim'] + 1)
        ]

    elif config['eval_mode'] == 'pos_intervention':
        metrics['pos_intervention_accs'] = [
            test_interventions(model, test_loader, PositiveIntervention, n)
            for n in range(train_result.config['concept_dim'] + 1)
        ]

    elif config['eval_mode'] == 'random_concepts':
        metrics['random_concept_acc'] = test_random_concepts(
            model, test_loader, concept_dim)
    
    elif config['eval_mode'] == 'random_residual':
        metrics['random_residual_acc'] = test_random_residual(
            model, test_loader, concept_dim)

    # Report evaluation metrics
    ray.train.report(metrics)



if __name__ == '__main__':
    disable_ray_storage_context()

    MODES = [
        'accuracy',
        'neg_intervention',
        'pos_intervention',
        'random_concepts',
        'random_residual',
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-dir', type=str, help='Experiment directory')
    parser.add_argument(
        '--mode', nargs='+', default=MODES, help='Evaluation modes')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for model inference')
    parser.add_argument(
        '--num-gpus', type=float, default=1, help='Number of GPUs to use (per model)')
    parser.add_argument(
        '--best-only', action='store_true',
        help='Only evaluate best trial result per experiment group')
    parser.add_argument(
        '--groupby', nargs='+', default=['dataset', 'model_type'],
        help='Config keys to group by when selecting best trial results'
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob('**/train/tuner.pkl')
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load train results
    results = load_train_results(
        experiment_path, best_only=args.best_only, groupby=args.groupby)

    # Create evaluation configs
    eval_configs = [
        {
            'train_result': result,
            'eval_mode': mode,
            'device': args.device,
        }
        for result in results
        for mode in args.mode
    ]

    # Run evaluations
    tuner = tune.Tuner(
        tune.with_resources(
            evaluate,
            resources={
                'cpu': 1,
                'gpu': args.num_gpus if torch.cuda.is_available() else 0
            },
        ),
        param_space=tune.grid_search(eval_configs),
        run_config=air.RunConfig(name='eval', storage_path=experiment_path),
    )
    eval_results = tuner.fit()
