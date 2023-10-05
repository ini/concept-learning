import argparse
import importlib
import torch
import torch.nn as nn
import ray.train

from copy import deepcopy
from pathlib import Path
from ray import air, tune
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Sequence

from loader import get_data_loaders
from models import ConceptModel, ConceptBottleneckModel, ConceptWhiteningModel
from train import concept_model_predict_fn, train
from utils import accuracy, disable_ray_storage_context



### Helper Modules

class Extend(nn.Sequential):

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


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
        if self.model_type == ConceptWhiteningModel:
            incorrect_concepts = 1 - 2 * concepts   # concept activations
        else:
            incorrect_concepts = 1 - concepts   # concepts values

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
            concepts = 2 * concepts - 1
        
        concept_dim = concepts.shape[-1]
        intervention_idx = torch.randperm(concept_dim)[:self.num_interventions]
        x[:, intervention_idx] = concepts[:, intervention_idx]
        return x


class Randomize(nn.Module):
    """
    Shuffle data along the batch dimension.
    """

    def __init__(self, idx: slice | Sequence[int] = slice(None)):
        super().__init__()
        self.idx = idx

    def forward(self, x: Tensor, **kwargs):
        order = torch.randperm(x.shape[0])
        x[..., self.idx] = x[order][..., self.idx]
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

    if isinstance(model, ConceptBottleneckModel):
        new_model.concept_network = Extend(
            new_model.concept_network,
            intervention_cls(num_interventions, type(model)),
        )
    elif isinstance(model, ConceptWhiteningModel):
        new_model.bottleneck_layer = Extend(
            new_model.bottleneck_layer,
            intervention_cls(num_interventions, type(model)),
        )

    return accuracy(new_model, test_loader, predict_fn=concept_model_predict_fn)

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

    if isinstance(model, ConceptBottleneckModel):
        new_model.concept_network = Extend(
            new_model.concept_network, Randomize())
    elif isinstance(model, ConceptWhiteningModel):
        new_model.bottleneck_layer = Extend(
            new_model.bottleneck_layer, Randomize(idx=slice(concept_dim)))

    return accuracy(new_model, test_loader, predict_fn=concept_model_predict_fn)

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

    if isinstance(model, ConceptBottleneckModel):
        new_model.residual_network = Extend(
            new_model.residual_network, Randomize())
    elif isinstance(model, ConceptWhiteningModel):
        new_model.bottleneck_layer = Extend(
            new_model.bottleneck_layer, Randomize(idx=slice(concept_dim)))

    return accuracy(new_model, test_loader, predict_fn=concept_model_predict_fn)



### Loading & Execution

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
    result = config['result']
    metrics = {}

    # Get data loaders
    _, _, test_loader, concept_dim, _ = get_data_loaders(
        result.config['dataset'],
        data_dir=result.config['data_dir'],
        batch_size=result.config['batch_size'],
    )

    # Load model
    model = load_model(result).to(config['device'])

    # Evaluate model
    if config['mode'] == 'accuracy':
        metrics['test_acc'] = accuracy(
            model, test_loader, predict_fn=concept_model_predict_fn)

    if config['mode'] == 'neg_intervention':
        metrics['neg_intervention_acc'] = test_interventions(
            model, test_loader, NegativeIntervention, config['num_interventions'])

    elif config['mode'] == 'pos_intervention':
        metrics['pos_intervention_acc'] = test_interventions(
            model, test_loader, PositiveIntervention, config['num_interventions'])

    elif config['mode'] == 'random_concepts':
        metrics['random_concept_acc'] = test_random_concepts(
            model, test_loader, concept_dim)
    
    elif config['mode'] == 'random_residual':
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
        '--modes', nargs='+', default=MODES, help='Evaluation modes')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for model inference',
    )
    parser.add_argument(
        '--num-gpus', type=float, default=1, help='Number of GPUs to use (per model)')
    args = parser.parse_args()

    # Get best trial results
    best_results = []
    experiment_path = Path(args.exp_dir).resolve()
    group_paths = [path.parent for path in experiment_path.glob('./train/**/tuner.pkl')]
    for path in group_paths:
        train_tuner = tune.Tuner.restore(str(path), trainable=train)
        results = train_tuner.get_results()
        best_result = results.get_best_result(metric='val_acc', mode='max')
        best_results.append(best_result)

    # Create evaluation configs
    eval_configs = [
        {
            'result': result,
            'mode': mode,
            'num_interventions': num_interventions,
            'device': args.device,
        }
        for result in best_results
        for mode in args.modes
        for num_interventions in range(result.config['concept_dim'] + 1)
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
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=air.RunConfig(name='eval', storage_path=experiment_path),
    )
    eval_results = tuner.fit()
