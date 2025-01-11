from __future__ import annotations
import os
import copy
import json

# make sure slurm isn't exposed
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]

if "SLURM_JOB_NAME" in os.environ:
    del os.environ["SLURM_JOB_NAME"]
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
import argparse
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import ray.train

from copy import deepcopy
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray import air, tune
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import DATASET_INFO
from lightning_ray import LightningTuner
from nn_extensions import Chain
from models import ConceptLightningModel
from models.mutual_info import MutualInformationLoss
from models.posthoc_concept_pred import (
    ConceptResidualConceptPred,
    ConceptEmbeddingConceptPred,
)
from train import make_concept_model, make_datamodule
from utils import cross_correlation, set_cuda_visible_devices


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
            concepts = 1 - concepts  # flip binary concepts to opposite values

        concept_dim = concepts.shape[-1]
        idx = torch.randperm(concept_dim)[: self.num_interventions]
        x[..., idx] = concepts[..., idx]
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
        accelerator="cpu" if MPSAccelerator.is_available() else "auto",
        enable_progress_bar=False,
    )
    return trainer.test(model, loader)[0]


def test_interventions(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    concept_dim: int,
    negative: bool,
    max_samples: int = 10,
) -> float:
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
    x = np.linspace(
        0, concept_dim + 1, num=min(concept_dim + 2, max_samples), dtype=int
    )
    # x = x[::-1]
    y = np.zeros(len(x))
    for i, num_interventions in enumerate(x):
        # intervention = Intervention(num_interventions, negative=negative)
        new_model = deepcopy(model)
        new_model.num_test_interventions = num_interventions
        new_model.concept_model.negative_intervention = negative

        # new_model.concept_model.target_network = Chain(
        #     intervention,
        #     new_model.concept_model.target_network,
        # )
        results = test(new_model, test_loader)
        y[i] = results["test_acc"]

    return {"x": x, "y": y}


def test_random_concepts(
    model: ConceptLightningModel, test_loader: DataLoader
) -> float:
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
        new_model.concept_model.concept_network, Randomize()
    )
    if hasattr(new_model.concept_model, "concept_prob_generators"):
        new_generators = nn.ModuleList()
        for generator in new_model.concept_model.concept_prob_generators:
            new_chain = Chain(generator, Randomize())
            new_generators.append(new_chain)
        new_model.concept_model.concept_prob_generators = new_generators
    results = test(new_model, test_loader)
    return results["test_acc"]


def test_random_residual(
    model: ConceptLightningModel, test_loader: DataLoader
) -> float:
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
        new_model.concept_model.residual_network, Randomize()
    )
    # self.concept_prob_generators = concept_network
    # self.concept_context_generators = residual_network
    if hasattr(new_model.concept_model, "concept_context_generators"):
        new_generators = nn.ModuleList()
        for generator in new_model.concept_model.concept_context_generators:
            new_chain = Chain(generator, Randomize())
            new_generators.append(new_chain)
        new_model.concept_model.concept_context_generators = new_generators
    results = test(new_model, test_loader)
    return results["test_acc"]


def test_correlation(model: ConceptLightningModel, test_loader: DataLoader) -> float:
    """
    Test mean absolute cross correlation between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    correlations = []
    for (data, concepts), target in test_loader:
        with torch.no_grad():
            _, residual, _ = model(data, concepts=concepts)
        correlations.append(cross_correlation(concepts, residual).abs().mean().item())

    return np.mean(correlations)


def test_mutual_info(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    num_mi_epochs: int = 5,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_mi_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    (data, concepts), targets = next(iter(test_loader))
    _, residual, _ = model(data, concepts=concepts)
    concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
    mutual_info_estimator = MutualInformationLoss(residual_dim, concept_dim)

    # Learn mutual information estimator
    for epoch in range(num_mi_epochs):
        for (data, concepts), targets in test_loader:
            with torch.no_grad():
                _, residual, _ = model(data, concepts=concepts)
            mutual_info_estimator.step(residual, concepts)

    # Calculate mutual information
    mutual_infos = []
    for (data, concepts), target in test_loader:
        with torch.no_grad():
            _, residual, _ = model(data, concepts=concepts)
        mutual_infos.append(mutual_info_estimator(residual, concepts).item())

    return np.mean(mutual_infos)


def test_concept_pred(
    model: ConceptLightningModel,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_train_epochs: int = 1,
    dataset=None,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_train_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    if dataset == "celeba":
        hidden_concepts = 0
    else:
        hidden_concepts = 0
    (data, concepts), targets = next(iter(test_loader))
    if hidden_concepts != 0:
        _, residual, _ = model(data, concepts=concepts[:, :-hidden_concepts])
    else:
        _, residual, _ = model(data, concepts=concepts)
    if model_type == "cem" or model_type == "cem_mi":
        concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1] // 2
        concept_predictor = ConceptEmbeddingConceptPred(
            residual_dim,
            concept_dim - hidden_concepts,
            binary=model.concept_model.concept_type == "binary",
            hidden_concept=hidden_concepts > 0,
            num_hidden_concept=hidden_concepts,
        )
    else:
        concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
        concept_predictor = ConceptResidualConceptPred(
            residual_dim,
            concept_dim,
            binary=model.concept_model.concept_type == "binary",
            hidden_concept=hidden_concepts > 0,
            num_hidden_concept=hidden_concepts,
        )

    best_val_loss = float("inf")
    best_predictor_state = None

    # Train the concept predictor
    for epoch in range(num_train_epochs):
        # Training phase
        concept_predictor.train()
        for (data, concepts), targets in train_loader:
            if model_type == "cem" or model_type == "cem_mi":
                with torch.no_grad():
                    if hidden_concepts != 0:
                        pre_contexts, residual, _ = model(
                            data, concepts=concepts[:, :-hidden_concepts]
                        )
                    else:
                        pre_contexts, residual, _ = model(data, concepts=concepts)
                contexts = pre_contexts.sigmoid()
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                x = pos_embedding * torch.unsqueeze(
                    contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                concept_predictor.step(x.detach(), concepts.detach())
            else:
                with torch.no_grad():
                    if hidden_concepts != 0:
                        pre_contexts, residual, _ = model(
                            data, concepts=concepts[:, :-hidden_concepts]
                        )
                    else:
                        _, residual, _ = model(data, concepts=concepts)
                concept_predictor.step(residual.detach(), concepts.detach())

        # Validation phase
        val_losses = []
        concept_predictor.eval()
        for (data, concepts), targets in val_loader:
            with torch.no_grad():
                if hidden_concepts != 0:
                    pre_contexts, residual, _ = model(
                        data, concepts=concepts[:, :-hidden_concepts]
                    )
                else:
                    pre_contexts, residual, _ = model(data, concepts=concepts)
                if model_type == "cem" or model_type == "cem_mi":
                    contexts = pre_contexts.sigmoid()
                    r_dim = residual.shape[-1]
                    pos_embedding = residual[:, :, : r_dim // 2]
                    neg_embedding = residual[:, :, r_dim // 2 :]
                    x = pos_embedding * torch.unsqueeze(
                        contexts, dim=-1
                    ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                else:
                    x = residual
                y_pred = concept_predictor(x)
                if model.concept_model.concept_type == "binary":
                    loss_fn = nn.BCEWithLogitsLoss()
                else:
                    loss_fn = nn.MSELoss()
                val_loss = loss_fn(y_pred, concepts).item()
                val_losses.append(val_loss)

        mean_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}: Validation loss = {mean_val_loss}")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_predictor_state = concept_predictor.state_dict()

    # Load the best predictor state
    if best_predictor_state is not None:
        concept_predictor.load_state_dict(best_predictor_state)

    # Evaluate the concept predictor
    metrics = []
    for i in range(concept_dim):
        metrics.append([])
    intchange_metrics = []
    for i in range(concept_dim):
        intchange_metrics.append([])

    for (data, concepts), target in test_loader:
        with torch.no_grad():
            if hidden_concepts != 0:
                pre_contexts, residual, _ = model(
                    data, concepts=concepts[:, :-hidden_concepts]
                )
            else:
                pre_contexts, residual, _ = model(data, concepts=concepts)
            if model_type == "cem" or model_type == "cem_mi":
                contexts = pre_contexts.sigmoid()
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                x_test = pos_embedding * torch.unsqueeze(
                    contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
            else:
                x_test = residual
            y_pred_base = concept_predictor(x_test)
            if model.concept_model.concept_type == "binary":
                y_pred_base = torch.sigmoid(y_pred_base)
                for i in range(concept_dim):
                    pred = (y_pred_base[:, i] > 0.5).float()
                    accuracy = (pred == concepts[:, i]).float().mean().item()
                    metrics[i].append(accuracy)
            else:
                for i in range(concept_dim):
                    mse = ((y_pred_base[:, i] - concepts[:, i]) ** 2).mean().item()
                    metrics[i].append(mse)

            # perform concept interventions with concept full concepts
            if model_type == "cem" or model_type == "cem_mi":
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                if hidden_concepts != 0:
                    x_test = pos_embedding * torch.unsqueeze(
                        concepts[:, :-hidden_concepts], dim=-1
                    ) + neg_embedding * torch.unsqueeze(
                        1 - concepts[:, :-hidden_concepts], dim=-1
                    )
                else:
                    x_test = pos_embedding * torch.unsqueeze(
                        concepts, dim=-1
                    ) + neg_embedding * torch.unsqueeze(1 - concepts, dim=-1)
            else:
                x_test = residual

            y_pred_intervention = concept_predictor(x_test)

            if model.concept_model.concept_type == "binary":
                y_pred_intervention = torch.sigmoid(y_pred_intervention)
                for i in range(concept_dim):
                    pred_intervene = (y_pred_intervention[:, i] > 0.5).float()
                    pred = (y_pred_base[:, i] > 0.5).float()
                    change = (pred != pred_intervene).float().mean().item()
                    intchange_metrics[i].append(change)

    # Calculate mean metric for each concept
    mean_metrics = np.array([np.mean(metric) for metric in metrics])
    mean_change_metrics = np.array([np.mean(metric) for metric in intchange_metrics])

    if hidden_concepts > 0:
        return np.array(
            [
                np.mean(mean_metrics[:-hidden_concepts]),
                np.mean(mean_metrics[-hidden_concepts:]),
                np.mean(mean_change_metrics[:-hidden_concepts]),
                np.mean(mean_change_metrics[-hidden_concepts:]),
            ]
        )
    else:
        return np.array(
            [
                np.mean(mean_metrics),
                0,
                np.mean(mean_change_metrics),
                0,
            ]
        )


def test_concept_change_probe(
    model: ConceptLightningModel,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_train_epochs: int = 1,
    dataset=None,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_train_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    if dataset == "celeba":
        hidden_concepts = 2
    else:
        hidden_concepts = 0
    (data, concepts), targets = next(iter(test_loader))
    if hidden_concepts != 0:
        _, residual, _ = model(data, concepts=concepts[:, :-hidden_concepts])
    else:
        _, residual, _ = model(data, concepts=concepts)
    if model_type == "cem" or model_type == "cem_mi":
        concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1] // 2
        concept_predictor = ConceptResidualConceptPred(
            (concept_dim - hidden_concepts) * residual_dim,
            concept_dim,
            binary=model.concept_model.concept_type == "binary",
            hidden_concept=hidden_concepts > 0,
            num_hidden_concept=hidden_concepts,
        )
    else:
        concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
        concept_predictor = ConceptResidualConceptPred(
            (concept_dim - hidden_concepts) + residual_dim,
            concept_dim,
            binary=model.concept_model.concept_type == "binary",
            hidden_concept=hidden_concepts > 0,
            num_hidden_concept=hidden_concepts,
        )

    best_val_loss = float("inf")
    best_predictor_state = None

    # Train the concept predictor
    for epoch in range(num_train_epochs):
        # Training phase
        concept_predictor.train()
        for (data, concepts), targets in train_loader:
            if model_type == "cem" or model_type == "cem_mi":
                with torch.no_grad():
                    if hidden_concepts != 0:
                        pre_contexts, residual, _ = model(
                            data, concepts=concepts[:, :-hidden_concepts]
                        )
                    else:
                        pre_contexts, residual, _ = model(data, concepts=concepts)
                contexts = pre_contexts.sigmoid()
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                x = pos_embedding * torch.unsqueeze(
                    contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                x = x.reshape((x.shape[0], -1))
                concept_predictor.step(x.detach(), concepts.detach())
            else:
                with torch.no_grad():
                    if hidden_concepts != 0:
                        pre_contexts, residual, _ = model(
                            data, concepts=concepts[:, :-hidden_concepts]
                        )
                    else:
                        pre_contexts, residual, _ = model(data, concepts=concepts)
                x = torch.cat((residual, pre_contexts), dim=-1)
                concept_predictor.step(x.detach(), concepts.detach())

        # Validation phase
        val_losses = []
        concept_predictor.eval()
        for (data, concepts), targets in val_loader:
            with torch.no_grad():
                if hidden_concepts != 0:
                    pre_contexts, residual, _ = model(
                        data, concepts=concepts[:, :-hidden_concepts]
                    )
                else:
                    pre_contexts, residual, _ = model(data, concepts=concepts)
                if model_type == "cem" or model_type == "cem_mi":
                    contexts = pre_contexts.sigmoid()
                    r_dim = residual.shape[-1]
                    pos_embedding = residual[:, :, : r_dim // 2]
                    neg_embedding = residual[:, :, r_dim // 2 :]
                    x = pos_embedding * torch.unsqueeze(
                        contexts, dim=-1
                    ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                    x = x.reshape((x.shape[0], -1))
                else:
                    x = torch.cat((residual, pre_contexts), dim=-1)

                y_pred = concept_predictor(x)
                if model.concept_model.concept_type == "binary":
                    loss_fn = nn.BCEWithLogitsLoss()
                else:
                    loss_fn = nn.MSELoss()
                val_loss = loss_fn(y_pred, concepts).item()
                val_losses.append(val_loss)

        mean_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}: Validation loss = {mean_val_loss}")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_predictor_state = concept_predictor.state_dict()

    # Load the best predictor state
    if best_predictor_state is not None:
        concept_predictor.load_state_dict(best_predictor_state)

    # Evaluate the concept predictor
    metrics = []
    for i in range(concept_dim):
        metrics.append([])

    num_changed_concepts_list = []
    concept_updated_list = []
    hidden_concepts_updated_list = []

    for (data, concepts), target in test_loader:
        with torch.no_grad():
            if hidden_concepts != 0:
                model.num_test_interventions = 1
                tup, int_idxs = model.forward_intervention(
                    ((data, concepts[:, :-hidden_concepts]), target),
                    0,
                    return_intervention_idxs=True,
                )
                pre_contexts, residual, _ = tup
                contexts = pre_contexts.sigmoid()
                intervened_contexts = (
                    contexts.detach() * (1 - int_idxs)
                    + concepts[:, :-hidden_concepts] * int_idxs
                )
            else:
                model.num_test_interventions = 1
                tup, int_idxs = model.forward_intervention(
                    ((data, concepts), target), 0, return_intervention_idxs=True
                )
                pre_contexts, residual, _ = tup
                contexts = pre_contexts.sigmoid()
                intervened_contexts = (
                    contexts.detach() * (1 - int_idxs) + concepts * int_idxs
                )

            if model_type == "cem" or model_type == "cem_mi":
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                x_test = pos_embedding * torch.unsqueeze(
                    contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                x_test = x_test.reshape((x_test.shape[0], -1))
            else:
                x_test = torch.cat((residual, contexts), dim=-1)
            y_pred_base = concept_predictor(x_test)

            if model.concept_model.concept_type == "binary":
                y_pred_base = torch.sigmoid(y_pred_base)
                for i in range(concept_dim):
                    pred = (y_pred_base[:, i] > 0.5).float()
                    accuracy = (pred == concepts[:, i]).float().mean().item()
                    metrics[i].append(accuracy)
            else:
                for i in range(concept_dim):
                    mse = ((y_pred_base[:, i] - concepts[:, i]) ** 2).mean().item()
                    metrics[i].append(mse)

            # perform concept interventions with concept full concepts
            if model_type == "cem" or model_type == "cem_mi":
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]

                x_test = pos_embedding * torch.unsqueeze(
                    intervened_contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - intervened_contexts, dim=-1)
                x_test = x_test.reshape((x_test.shape[0], -1))

            else:
                x_test = torch.cat((residual, intervened_contexts), dim=-1)

            y_pred_intervention = concept_predictor(x_test)
            if model.concept_model.concept_type == "binary":
                y_pred_intervention = torch.sigmoid(y_pred_intervention)

            pred_concepts = np.array(y_pred_base >= 0.5)
            pred_int_concepts = np.array(y_pred_intervention >= 0.5)
            np_concepts = np.array(concepts)

            # Vectorized calculations
            mask = np.array(
                int_idxs == 0
            )  # Assuming int_idxs is of shape (batch_size, 6)
            # mask = np.pad(
            #     mask, ((0, 0), (0, 2)), "constant", constant_values=0
            # )  # Add buffer of 0's to the right to make it (batch_size, 8)
            # assert (
            #     0
            # ), f"{int_idxs.shape} {pred_concepts.shape} {pred_int_concepts.shape} {concepts.shape}"
            if dataset == "celeba":
                num_changed_concepts = np.sum(
                    (pred_concepts[:, :6] != pred_int_concepts[:, :6]) & ~mask, axis=1
                )
                concept_updated = np.any(
                    np_concepts[:, :6] != pred_int_concepts[:, :6] & mask, axis=1
                )
                hidden_concepts_updated = np.sum(
                    pred_concepts[:, 6:8] != pred_int_concepts[:, 6:8], axis=1
                )
            else:
                num_changed_concepts = np.sum(
                    (pred_concepts != pred_int_concepts) & ~mask, axis=1
                )
                concept_updated = np.any(
                    np_concepts != pred_int_concepts & mask, axis=1
                )
                hidden_concepts_updated = [0.0]

            num_changed_concepts_list.extend(num_changed_concepts)
            concept_updated_list.extend(concept_updated)
            hidden_concepts_updated_list.extend(hidden_concepts_updated)

        # assert (
        #     0
        # ), f"{gt_concepts[9]} and {pred_concepts[9]} and {pred_int_concepts[9]} and {int_idxs.shape}"

    # Calculate mean metrics
    mean_accuracy = np.array([np.mean(metric) for metric in metrics])
    mean_num_changed_concepts = np.mean(num_changed_concepts_list)
    mean_concept_updated = np.mean(concept_updated_list)
    mean_hidden_concepts_updated = np.mean(hidden_concepts_updated_list)
    return (
        mean_accuracy,
        mean_num_changed_concepts,
        mean_concept_updated,
        mean_hidden_concepts_updated,
    )


def test_concept_change(
    model: ConceptLightningModel,
    model_type: str,
    test_loader: DataLoader,
    dataset=None,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_mi_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    with open("/home/renos/label_invert.json", "r") as f:
        label_invert = json.load(f)
    (data, concepts), targets = next(iter(test_loader))
    _, residual, _ = model(data, concepts=concepts)

    def invert_binarize(binary_int):
        binary_str = bin(binary_int)[2:].zfill(8)
        concepts = np.array([int(bit) for bit in binary_str], dtype=int)
        return concepts

    def update_all(vector):
        return np.array(
            [invert_binarize(int(label_invert[str(int(v))])) for v in vector]
        )

    num_changed_concepts_list = []
    concept_updated_list = []
    int_concept_correct_list = []
    base_concept_correct_list = []
    hidden_concepts_updated_list = []

    for (data, concepts), target in test_loader:
        with torch.no_grad():
            _, residual, y_pred = model(data, concepts=concepts)
            model.num_test_interventions = 1
            tup, int_idxs = model.forward_intervention(
                ((data, concepts), target), 0, return_intervention_idxs=True
            )
            _, _, y_pred_int = tup
        y_pred_amax = torch.argmax(y_pred, dim=1)
        y_pred_int_amex = torch.argmax(y_pred_int, dim=1)
        gt_concepts = update_all(target)
        pred_concepts = update_all(y_pred_amax)
        pred_int_concepts = update_all(y_pred_int_amex)

        # Vectorized calculations
        #only other concepts
        mask = np.array(int_idxs == 0)  # Assuming int_idxs is of shape (batch_size, 6)

        #number of supervised concepts changed during an intervention
        num_changed_concepts = np.sum(
            (pred_concepts[:, :6] != pred_int_concepts[:, :6]) & mask, axis=1
        )
        # Did an intervention change a concept?
        concept_updated = np.any(
            (pred_concepts[:, :6] != pred_int_concepts[:, :6]) & ~mask, axis=1
        )

        # Is concept correct after intervention?
        int_concept_correct = np.any(
            (gt_concepts[:, :6] == pred_int_concepts[:, :6]) & ~mask, axis=1
        )
        # Is concept correct before?
        base_concept_correct = np.any(
            (gt_concepts[:, :6] == pred_concepts[:, :6]) & ~mask, axis=1
        )
        hidden_concepts_updated = np.sum(
            pred_concepts[:, 6:8] != pred_int_concepts[:, 6:8], axis=1
        )

        concept_updated_list.extend(concept_updated)
        num_changed_concepts_list.extend(num_changed_concepts)
        int_concept_correct_list.extend(int_concept_correct)
        base_concept_correct_list.extend(base_concept_correct)
        hidden_concepts_updated_list.extend(hidden_concepts_updated)

        # assert (
        #     0
        # ), f"{gt_concepts[9]} and {pred_concepts[9]} and {pred_int_concepts[9]} and {int_idxs.shape}"
    num_changed_concepts = np.array(num_changed_concepts_list)
    concept_updated = np.array(concept_updated_list).astype(bool)

    int_concept_correct = np.array(int_concept_correct_list)
    base_concept_correct = np.array(base_concept_correct_list).astype(bool)
    hidden_concepts_updated = np.array(hidden_concepts_updated_list)
    base_concept_correct = np.array(base_concept_correct_list).astype(bool)


    # Calculate Metrics
    mean_num_changed_concepts = np.sum(num_changed_concepts & ~base_concept_correct) / np.sum(~base_concept_correct)
    mean_hidden_concepts_updated = np.sum(hidden_concepts_updated & ~base_concept_correct) / np.sum(~base_concept_correct)
    concept_updated_when_wrong = np.sum(concept_updated & ~base_concept_correct) / np.sum(~base_concept_correct)

    return mean_num_changed_concepts, concept_updated_when_wrong, mean_hidden_concepts_updated


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
        if config["model_type"] == "concept_whitening":
            if config["eval_mode"].endswith("intervention"):
                print("Interventions not supported for concept whitening models")
                continue

        if config["model_type"] == "no_residual" or config["residual_dim"] == 0:
            if config["eval_mode"] in ("correlation", "mutual_info"):
                print("Correlation / MI metrics not available for no-residual models")
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
    if config["eval_mode"] == "concept_change_probe" and config["dataset"] == "celeba":
        new_config = copy.deepcopy(config)
        new_config["num_concepts"] = 8
        train_loader = make_datamodule(**new_config).train_dataloader()
        val_loader = make_datamodule(**new_config).val_dataloader()
        test_loader = make_datamodule(**new_config).test_dataloader()
    else:
        train_loader = make_datamodule(**config).train_dataloader()
        val_loader = make_datamodule(**config).val_dataloader()
        test_loader = make_datamodule(**config).test_dataloader()

    # Load model
    tuner = LightningTuner("val_acc", "max")
    model = tuner.load_model(make_concept_model, config["train_result"])

    # Evaluate model
    if config["eval_mode"] == "accuracy":
        results = test(model, test_loader)
        for key in ("test_acc", "test_concept_acc"):
            if key in results:
                metrics[key] = results[key]

    elif config["eval_mode"] == "neg_intervention":
        concept_dim = DATASET_INFO[config["dataset"]]["concept_dim"]
        metrics["neg_intervention_accs"] = test_interventions(
            model, test_loader, concept_dim, negative=True
        )

    elif config["eval_mode"] == "pos_intervention":
        concept_dim = DATASET_INFO[config["dataset"]]["concept_dim"]
        metrics["pos_intervention_accs"] = test_interventions(
            model, test_loader, concept_dim, negative=False
        )

    elif config["eval_mode"] == "random_concepts":
        metrics["random_concept_acc"] = test_random_concepts(model, test_loader)

    elif config["eval_mode"] == "random_residual":
        metrics["random_residual_acc"] = test_random_residual(model, test_loader)

    elif config["eval_mode"] == "correlation":
        metrics["mean_abs_cross_correlation"] = test_correlation(model, test_loader)

    elif config["eval_mode"] == "mutual_info":
        metrics["mutual_info"] = test_mutual_info(model, test_loader)

    elif config["eval_mode"] == "concept_pred":
        metrics["concept_pred"] = test_concept_pred(
            model,
            config["model_type"],
            train_loader,
            val_loader,
            test_loader,
            dataset=config["dataset"],
        )
    elif config["eval_mode"] == "concept_change":
        metrics["concept_change"] = test_concept_change(
            model,
            config["model_type"],
            test_loader,
            dataset=config["dataset"],
        )
    elif config["eval_mode"] == "concept_change_probe":
        metrics["concept_change_probe"] = test_concept_change_probe(
            model,
            config["model_type"],
            train_loader,
            val_loader,
            test_loader,
            dataset=config["dataset"],
        )

    # Report evaluation metrics
    ray.train.report(metrics)


if __name__ == "__main__":
    MODES = [
        # "accuracy",
        # "neg_intervention",
        # "pos_intervention",
        # "random_concepts",
        # "random_residual",
        # "correlation",
        # "mutual_info",
        # "concept_pred",
        "concept_change",
        #"concept_change_probe",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        help="Experiment directory",
    )
    parser.add_argument("--mode", nargs="+", default=MODES, help="Evaluation modes")
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["dataset", "model_type"],
        help="Config keys to group by when selecting best trial results",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all trained models (instead of best trial per group)",
    )
    parser.add_argument(
        "--num-cpus", type=float, default=1, help="Number of CPUs to use (per model)"
    )
    parser.add_argument(
        "--num-gpus", type=float, default=1, help="Number of GPUs to use (per model)"
    )
    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob("**/train/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load train results
    train_folder = "train"
    print("Loading training results from", experiment_path / train_folder)
    tuner = LightningTuner.restore(experiment_path / train_folder)
    if args.all:
        results = tuner.get_results()
    else:
        results = [
            group.get_best_result()
            for group in tuner.get_results(groupby=args.groupby).values()
        ]

    # Create evaluation configs
    results = [result for result in results if result.config is not None]
    eval_configs = filter_eval_configs(
        [
            {
                **result.config["train_loop_config"],
                "train_result": result,
                "eval_mode": mode,
            }
            for result in results
            for mode in args.mode
        ]
    )

    # Get available resources
    if args.num_gpus < 1:
        set_cuda_visible_devices(available_memory_threshold=args.num_gpus)

    # Run evaluations
    eval_folder = "eval"
    tuner = tune.Tuner(
        tune.with_resources(
            evaluate,
            resources={
                "cpu": args.num_cpus,
                "gpu": args.num_gpus if torch.cuda.is_available() else 0,
            },
        ),
        param_space=tune.grid_search(eval_configs),
        run_config=air.RunConfig(name=eval_folder, storage_path=experiment_path),
    )
    eval_results = tuner.fit()
