from __future__ import annotations

import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Iterable, Sequence

from club import CLUB
from models import ConceptBottleneckModel, ConceptWhiteningModel
from utils import (
    concept_model_accuracy,
    cross_correlation,
    to_device,
    Random,
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
        return [
            test_negative_interventions(
                model, test_loader, concept_dim, num_interventions=n)
            for n in tqdm(num_interventions)
        ]

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

    acc = concept_model_accuracy(new_model, data_loader)
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

    acc = concept_model_accuracy(new_model, data_loader)
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
