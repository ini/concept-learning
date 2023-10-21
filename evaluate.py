import argparse
import os
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
from torchmetrics import Accuracy
from typing import Sequence

from loader import get_data_loaders
from models import ConceptModel, ConceptWhiteningModel
from ray_utils import filter_results
from train import make_concept_model, get_ray_trainer

import nn_extensions # enable '+' operator for chaining modules



### Accuracy Metrics

def accuracy(model: ConceptModel, loader: DataLoader) -> float:
    """
    Compute model accuracy over the given data.

    Parameters
    ----------
    model : ConceptModel
        Model to evaluate
    loader : DataLoader
        Data loader
    """
    targets, target_preds = [], []
    for (x, c), y in loader:
        y_pred = model(x, concepts=c)[-1]
        num_classes = y_pred.shape[-1]
        target_preds.append(y_pred)
        targets.append(y)

    targets = torch.cat(targets, dim=0)
    target_preds = torch.cat(target_preds, dim=0)
    accuracy_fn = Accuracy(task='multiclass', num_classes=num_classes)
    return accuracy_fn(target_preds, targets).item()

def concept_accuracy(model: ConceptModel, loader: DataLoader) -> float:
    """
    Compute model concept accuracy over the given data.

    Parameters
    ----------
    model : ConceptModel
        Model to evaluate
    loader : DataLoader
        Data loader
    """
    concepts, concept_preds = [], []
    for (x, c), y in loader:
        c_pred = model(x, concepts=c)[0]
        concept_preds.append(c_pred)
        concepts.append(c)

    concepts = torch.cat(concepts, dim=0)
    concept_preds = torch.cat(concept_preds, dim=0)
    concept_accuracy_fn = Accuracy(task='binary')
    return concept_accuracy_fn(concept_preds, concepts).item()



### Helper Modules

class Subscriptable(type):
    """
    Subscriptable metaclass.
    Allows initialization of classes with subscripting (e.g. `x = MyClass[:10]`).
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
    return accuracy(new_model, test_loader)

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
    return accuracy(new_model, test_loader)

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
    return accuracy(new_model, test_loader)



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
    tuner = tune.Tuner.restore(str(results_path), trainable=get_ray_trainer())
    results = tuner.get_results()
    
    if not best_only:
        return list(results)

    # Group results by config keys in `groupby`
    trials = defaultdict(list)
    for trial in results._experiment_analysis.trials:
        group_key = tuple(trial.config[key] for key in groupby)
        trials[group_key].append(trial)

    # Get best result for each group
    return [
        filter_results(trials[group_key].__contains__, results).get_best_result()
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
    train_config = result.config['train_loop_config']
    checkpoint_dir = result.get_best_checkpoint('val_acc', 'max').path
    model = make_concept_model(**train_config).concept_model
    model_path = Path(checkpoint_dir) / 'model.pt'
    model.load_state_dict(torch.load(model_path))
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
    train_config = train_result.config['train_loop_config']
    metrics = {}

    # Get data loaders
    _, _, test_loader = get_data_loaders(
        train_config['dataset'], data_dir=train_config['data_dir'], batch_size=10000)

    # Load model
    model = load_model(train_result).to(config['device'])
    model.eval()

    # Get concept dim via dummy pass
    loader = DataLoader(test_loader.dataset, batch_size=1)
    (data, concepts), targets = next(iter(loader))
    concept_dim = model(data, concepts=concepts)[0].shape[-1]

    # Evaluate model
    if config['eval_mode'] == 'accuracy':
        metrics['test_acc'] = accuracy(model, test_loader)
        metrics['test_concept_acc'] = concept_accuracy(model, test_loader)

    if config['eval_mode'] == 'neg_intervention':
        metrics['neg_intervention_accs'] = [
            test_interventions(model, test_loader, NegativeIntervention, n)
            for n in range(concept_dim + 1)
        ]

    elif config['eval_mode'] == 'pos_intervention':
        metrics['pos_intervention_accs'] = [
            test_interventions(model, test_loader, PositiveIntervention, n)
            for n in range(concept_dim + 1)
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
