import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Iterable

from models import ConceptBottleneckModel, ConceptWhiteningModel
from utils import accuracy, to_device, train_multiclass_classification



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

def test_residual_to_label(
    model: ConceptBottleneckModel | ConceptWhiteningModel,
    test_loader: DataLoader,
    residual_dim: int,
    num_classes: int):
    """
    Test the accuracy of a model trained to predict the label from the residual only.

    Parameters
    ----------
    model : ConceptBottleneckModel or ConceptWhiteningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    residual_dim : int
        Dimension of the residual vector
    num_classes : int
        Number of label classes
    """
    model.eval()
    residual_to_label_model = nn.Sequential(
        nn.Linear(residual_dim, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, num_classes),
    ).to(next(model.parameters()).device)

    if isinstance(model, ConceptBottleneckModel):
        def preprocess_fn(batch):
            with torch.no_grad():
                (X, c), y = batch
                _, residual, _ = model(X)
                return residual, y
    elif isinstance(model, ConceptWhiteningModel):
        def preprocess_fn(batch):
            with torch.no_grad():
                (X, c), y = batch
                residual = model.activations(X)[:, c.shape[1]:]
                return residual, y

    print('Training ...')
    train_multiclass_classification(
        residual_to_label_model, test_loader, preprocess_fn=preprocess_fn)
    return accuracy(
        residual_to_label_model, test_loader, preprocess_fn=preprocess_fn)
