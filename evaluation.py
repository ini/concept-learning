from __future__ import annotations

import argparse
import importlib
import json
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Iterable, Sequence

from loader import get_data_loaders
from models import ConceptBottleneckModel, ConceptWhiteningModel
from utils import (
    accuracy,
    cross_correlation,
    to_device,
)



class Random(nn.Module):
    """
    Replaces input data with random noise.
    """

    def __init__(
        self, random_fn=torch.randn_like, indices: slice | Sequence[int] = slice(None)):
        """
        Parameters
        ----------
        random_fn : Callable(Tensor) -> Tensor
            Function to generate random noise
        indices : slice or Sequence[int]
            Feature indices to replace with random noise
        """
        super().__init__()
        self.random_fn = random_fn
        self.indices = indices

    def forward(self, x: Tensor):
        x[..., self.indices] = self.random_fn(x[..., self.indices])
        return x





def negative_intervention(
    model: ConceptBottleneckModel | ConceptWhiteningModel,
    concept_preds: Tensor,
    concepts: Tensor,
    num_interventions: int):
    """
    Update predictions for a random subset of concepts with incorrect concept values.

    Parameters
    ----------
    model : ConceptBottleneckModel or ConceptWhiteningModel
        Model to intervene on
    concept_preds : Tensor of shape (num_samples, concept_dim)
        Concept predictions
    concepts : Tensor of shape (num_samples, concept_dim)
        Ground-truth concept values
    num_interventions : int
        Number of concepts to intervene on
    """
    if isinstance(model, ConceptBottleneckModel):
        incorrect_concepts = 1 - concepts   # binary concepts
    elif isinstance(model, ConceptWhiteningModel):
        incorrect_concepts = 1 - 2 * concepts # concept activations

    intervention_idx = torch.randperm(concept_preds.shape[-1])[:num_interventions]
    concept_preds[:, intervention_idx] = incorrect_concepts[:, intervention_idx]
    return concept_preds

def test_positive_interventions(
    model: ConceptBottleneckModel | ConceptWhiteningModel,
    test_loader: DataLoader):

    device = next(model.parameters()).device
    model.eval()
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            (X, c), y = to_device(batch, device)

            if isinstance(model, ConceptBottleneckModel):
                _, _, original_preds = model(X)
                _, _, intervened_preds = model(X, concept_preds=c)

            elif isinstance(model, ConceptWhiteningModel):
                original_preds = model(X)
                activations = model.activations(X)
                activations[:, :c.shape[-1]] = c
                intervened_preds = model.target_network(activations)

            original_incorrect = (original_preds.argmax(-1) != y)
            intervened_correct = (intervened_preds.argmax(-1) == y)

            num_correct += (original_incorrect & intervened_correct).sum().item()
            num_samples += original_incorrect.sum().item()

        return num_correct / num_samples


def test_negative_interventions(
    model: ConceptBottleneckModel | ConceptWhiteningModel,
    test_loader: DataLoader,
    concept_dim: int,
    num_interventions: int | Iterable[int]) -> float | list[float]:
    """
    Test model accuracy with negative interventions.

    Parameters
    ----------
    model : ConceptBottleneckModel or ConceptWhiteningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Total number of concepts
    num_interventions : int or Iterable[int]
        Number of concepts to intervene on
    """
    if isinstance(num_interventions, Iterable):
        return np.array([
            test_negative_interventions(
                model, test_loader, concept_dim, num_interventions=n)
            for n in tqdm(num_interventions)
        ])

    device = next(model.parameters()).device
    model.eval()
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            (X, c), y = to_device(batch, device)

            if isinstance(model, ConceptBottleneckModel):
                X = model.base_network(X)
                concept_preds = model.concept_network(X)
                concept_preds = negative_intervention(
                    model, concept_preds, c, num_interventions)
                _, _, target_preds = model(X, concept_preds=concept_preds)

            elif isinstance(model, ConceptWhiteningModel):
                X = model.base_network(X)
                bottleneck = X
                while bottleneck.ndim < 4:
                    bottleneck = bottleneck.unsqueeze(-1)
                bottleneck = model.bottleneck_layer(bottleneck).view(X.shape)
                bottleneck[:, :concept_dim] = negative_intervention(
                    model, bottleneck[:, :concept_dim], c, num_interventions)
                target_preds = model.target_network(bottleneck)

            prediction = target_preds.argmax(-1)
            num_correct += (prediction == y).sum().item()
            num_samples += y.size(0)

        return num_correct / num_samples

def test_random_concepts(
    model: ConceptBottleneckModel | ConceptWhiteningModel,
    data_loader: DataLoader,
    residual_dim: int) -> float:
    """
    Test the accuracy of a model with random concept values.

    Parameters
    ----------
    model : ConceptBottleneckModel or ConceptWhiteningModel
        Model to evaluate
    data_loader : DataLoader
        Train data loader
    residual_dim : int
        Dimension of the residual vector
    """
    class Invert(nn.Module):
        def forward(self, x):
            return 1 - x

    new_model = deepcopy(model)
    if isinstance(model, ConceptBottleneckModel):
        new_model.concept_network = nn.Sequential(
            new_model.concept_network,
            #Invert(),
            Random(random_fn=lambda x: torch.randint_like(x, 2)),
        )
    elif isinstance(model, ConceptWhiteningModel):
        new_model.bottleneck_layer = nn.Sequential(
            new_model.bottleneck_layer,
            Random(indices=slice(residual_dim)),
        )

    acc = accuracy(new_model, data_loader)
    print('Test Classification Accuracy (Random Concepts):', acc)
    return acc

def test_random_residual(
    model: ConceptBottleneckModel | ConceptWhiteningModel,
    data_loader: DataLoader,
    residual_dim: int) -> float:
    """
    Test the accuracy of a model with random residual.

    Parameters
    ----------
    model : ConceptBottleneckModel or ConceptWhiteningModel
        Model to evaluate
    data_loader : DataLoader
        Data to evaluate on
    residual_dim : int
        Dimension of the residual vector
    """
    new_model = deepcopy(model)
    if isinstance(model, ConceptBottleneckModel):
        new_model.residual_network = nn.Sequential(
            new_model.residual_network, Random())
    elif isinstance(model, ConceptWhiteningModel):
        new_model.bottleneck_layer = nn.Sequential(
            new_model.bottleneck_layer, Random(indices=slice(-residual_dim, None)))

    acc = accuracy(new_model, data_loader)
    print('Test Classification Accuracy (Random Residual):', acc)
    return acc

def test_concept_residual_correlation(
    model: ConceptBottleneckModel | ConceptWhiteningModel,
    test_loader: DataLoader):
    """
    Test the cross-correlation between the residual and the concept values.
    """
    device = next(model.parameters()).device
    model.eval()
    correlations = []
    with torch.no_grad():
        for batch in test_loader:
            (X, concepts), y = to_device(batch, device)
            if isinstance(model, ConceptBottleneckModel):
                concept_preds, residual, _ = model(X)
            elif isinstance(model, ConceptWhiteningModel):
                activations = model.activations(X)
                concept_preds = activations[:, :concepts.shape[-1]]
                residual = model.activations(X)[:, concepts.shape[-1]:]

            R = cross_correlation(residual, concept_preds)
            correlations.append(R.abs().mean().item())

    return sum(correlations) / len(correlations)



def get_config(model_dir):
    with open(Path(model_dir) / 'params.json') as file:
        return json.load(file)

def get_concept_dim(config):
    _, _, _, concept_dim, _ = get_data_loaders(
        config['dataset'], data_dir=config['data_dir'], batch_size=1)
    return concept_dim



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modes', nargs='+', default=['pos_intervention', 'neg_intervention'],
        help='Evaluation modes')
    parser.add_argument(
        '--load_dir', type=Path, default='saved/pitfalls/2023-10-04_00_14_53',
        help='Experiment configuration module')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on')
    args = parser.parse_args()

    results = {mode: {} for mode in args.modes}
    model_dirs = [
        path for path in args.load_dir.iterdir() if (path / 'params.json').exists()]

    for model_dir in model_dirs:
        config = get_config(model_dir)
        experiment_module = importlib.import_module(config['experiment_module_name'])

        # Load data
        _, _, test_loader, concept_dim, num_classes = get_data_loaders(
            config['dataset'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size']
        )

        # Update config with dataset information
        config['concept_dim'] = concept_dim
        config['num_classes'] = num_classes

        # Create model
        make_bottleneck_model = experiment_module.make_bottleneck_model
        make_whitening_model = experiment_module.make_whitening_model
        if config['model_type'] == 'no_residual':
            model = make_bottleneck_model(dict(config, residual_dim=0)).to(args.device)
        elif config['model_type'] == 'whitened_residual':
            model = make_whitening_model(config).to(args.device)
        else:
            model = make_bottleneck_model(config).to(args.device)

        # Load trained model parameters
        model.load_state_dict(torch.load(model_dir / 'model.pt'))

        # Evaluate model
        for mode in args.modes:
            if mode == 'pos_intervention':
                results[mode][model_dir] = test_positive_interventions(
                    model, test_loader)
            elif mode == 'neg_intervention':
                results[mode][model_dir] = test_negative_interventions(
                    model, test_loader, concept_dim,
                    num_interventions=range(0, concept_dim + 1),
                )



    ### Plotting

    import matplotlib.pyplot as plt
    from collections import defaultdict

    for mode in args.modes:
        if mode == 'pos_intervention':
            # Group results by dataset
            results_by_dataset = defaultdict(dict)
            for model_dir in results['pos_intervention']:
                config = get_config(model_dir)
                results_by_dataset[config['dataset']][model_dir] = results[mode][model_dir]

            # Plot results for each dataset
            for dataset_name in results_by_dataset:
                x, y = [], []
                for model_dir in sorted(results_by_dataset[dataset_name].keys()):
                    config = get_config(model_dir)
                    x.append(config['model_type'])
                    y.append(results_by_dataset[dataset_name][model_dir])
                    plt.bar(x, y)

                plt.ylabel('Re-Classification Accuracy')
                plt.title(
                    f'Positive Interventions: {dataset_name.replace("_", " ").title()}')
                plt.show()

        elif mode == 'neg_intervention':
            # Group results by dataset
            results_by_dataset = defaultdict(dict)
            for model_dir in results['pos_intervention']:
                config = get_config(model_dir)
                results_by_dataset[config['dataset']][model_dir] = results[mode][model_dir]

            # Plot results for each dataset
            for dataset_name in results_by_dataset:
                for model_dir in sorted(results_by_dataset[dataset_name].keys()):
                    config = get_config(model_dir)
                    num_interventions = np.arange(get_concept_dim(config) + 1)
                    accuracies = results_by_dataset[dataset_name][model_dir]
                    plt.plot(num_interventions, 1 - accuracies, label=config['model_type'])

                plt.xlabel('# of Concepts Intervened')
                plt.ylabel('Classification Error')
                plt.title(
                    f'Negative Interventions: {dataset_name.replace("_", " ").title()}')
                plt.legend()
                plt.show()
