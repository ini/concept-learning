from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from typing import Any, Callable, Iterable, Literal

from nn_extensions import VariableKwargs
from utils import accuracy, unwrap



### Typing

ConceptBatch = tuple[tuple[Tensor, Tensor], Tensor]  # ((data, concepts), targets)



### OAI Loss

def oai_loss_fn(
    model: ConceptLightningModel,
    batch: ConceptBatch,
    outputs: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    (data, concepts), targets = batch
    concept_logits, residual, target_logits = outputs

    assert isinstance(concepts, list) and len(concepts) == 3

    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)
    y = targets
    C_feats, C_feats_not_nan, C_feats_loss_class_wts = concepts

    # Parse outputs from deep model
    C_hat = concept_logits  # outputs['C']
    y_hat = target_logits  # outputs['y']

    # Loss for y
    # if self.y_loss_type == "reg":
    #     loss_y = nn.MSELoss()(input=y_hat, target=y)
    # elif self.y_loss_type == "cls":
    classes_per_y_col = [4]
    C_loss_type = "reg"
    # they use regression so this effectively does nothing
    classes_per_C_col = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    C_loss_weigh_class = False
    additional_loss_weighting = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    y_long = y.long()
    loss_y = []
    start_id = 0
    for i, N_cls in enumerate(classes_per_y_col):
        loss_y.append(
            nn.CrossEntropyLoss(reduction="none")(
                y_hat[:, start_id : start_id + N_cls], y_long[:, i]
            )
        )
        start_id += N_cls
        loss_y = torch.stack(loss_y, dim=1)
        loss_y = loss_y.sum(dim=1).mean(dim=0)
        assert start_id == sum(classes_per_y_col)

        # Loss for C
        if C_loss_type == "reg":
            loss_C = (C_feats - C_hat) ** 2
        elif C_loss_type == "cls":
            C_feats_long = C_feats.long()
            loss_C = []
            start_id = 0
            for i, N_cls in enumerate(classes_per_C_col):
                loss_C.append(
                    nn.CrossEntropyLoss(reduction="none")(
                        C_hat[:, start_id : start_id + N_cls], C_feats_long[:, i]
                    )
                )
                start_id += N_cls
            loss_C = torch.stack(loss_C, dim=1)
            assert start_id == sum(classes_per_C_col)

    # Compute loss only if feature is not NaN.
    loss_C = loss_C * C_feats_not_nan
    # We upweigh rare classes (within each concept)
    # to allow the model to pay attention to it.
    if C_loss_weigh_class:
        loss_class_wts = C_feats_loss_class_wts
        loss_C *= loss_class_wts.float().cuda()
    loss_C *= torch.FloatTensor([additional_loss_weighting]).cuda()
    concept_loss = loss_C.sum(dim=1).mean(dim=0) / (
        sum(additional_loss_weighting) + 1.0
    )
    target_loss = loss_y

    # # Final loss
    # loss = loss_y + loss_C
    # loss /= (sum(self.additional_loss_weighting) + 1.)

    model.log("concept_loss", concept_loss, **model.log_kwargs)

    # Residual loss
    concept_preds = model.concept_model.get_concept_predictions(concept_logits)
    residual_loss = model.residual_loss_fn(residual, concept_preds)
    if residual_loss.requires_grad:
        model.log("residual_loss", residual_loss, **model.log_kwargs)

    # Target loss
    if target_loss.requires_grad:
        model.log("target_loss", target_loss, **model.log_kwargs)

    return target_loss + (model.alpha * concept_loss) + (model.beta * residual_loss)

### Concept Models

class ConceptModel(nn.Module):
    """
    Base class for concept models.

    Attributes
    ----------
    base_network : nn.Module
        Network that pre-processes input
    concept_network : nn.Module
        Network that takes the output of `base_network` and
        generates concept logits
    residual_network : nn.Module
        Network that takes the output of `base_network` and
        generates a residual vector
    bottleneck_layer : nn.Module
        Network that post-processes the concatenated output of
        `concept_network` and `residual_network`
    target_network : nn.Module
        Network that takes the output of `bottleneck_layer` and
        generates target logits
    training_mode : one of {'independent', 'sequential', 'joint'}
        Training mode (see https://arxiv.org/abs/2007.04612)
    """

    def __init__(
        self,
        concept_network: nn.Module,
        residual_network: nn.Module,
        target_network: nn.Module,
        base_network: nn.Module = nn.Identity(),
        bottleneck_layer: nn.Module = nn.Identity(),
        concept_type: Literal['binary', 'continuous'] = 'binary',
        training_mode: Literal['independent', 'sequential', 'joint'] = 'independent',
        **kwargs,
    ):
        """
        Parameters
        ----------
        concept_network : nn.Module -> (..., concept_dim)
            Concept prediction network
        residual_network : nn.Module -> (..., residual_dim)
            Residual network
        target_network : nn.Module (..., bottleneck_dim) -> (..., num_classes)
            Target network
        base_network : nn.Module
            Optional base network
        bottleneck_layer : nn.Module (..., bottleneck_dim) -> (..., bottleneck_dim)
            Optional bottleneck layer
        concept_type : one of {'binary', 'continuous'}
            Concept type
        training_mode : one of {'independent', 'sequential', 'joint'}
            Training mode (see https://arxiv.org/abs/2007.04612)
        """
        super().__init__()
        self.base_network = VariableKwargs(base_network)
        self.concept_network = VariableKwargs(concept_network)
        self.residual_network = VariableKwargs(residual_network)
        self.target_network = VariableKwargs(target_network)
        self.bottleneck_layer = VariableKwargs(bottleneck_layer)
        self.concept_type = concept_type
        self.training_mode = training_mode

    def forward(
        self,
        x: Tensor,
        concepts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values

        Returns
        -------
        concept_logits : Tensor
            Concept logits
        residual : Tensor
            Residual vector
        target_logits : Tensor
            Target logits
        """
        concepts = concepts[0] if isinstance(concepts, (list, tuple)) else concepts

        # Get concept logits & residual
        x = self.base_network(x, concepts=concepts)
        concept_logits = self.concept_network(x, concepts=concepts)
        residual = self.residual_network(x, concepts=concepts)

        # Process concept logits & residual via bottleneck layer
        if not isinstance(unwrap(self.bottleneck_layer), nn.Identity):
            x = torch.cat([concept_logits, residual], dim=-1)
            x = self.bottleneck_layer(x, concepts=concepts)
            concept_dim, residual_dim = concept_logits.shape[-1], residual.shape[-1]
            concept_logits, residual = x.split([concept_dim, residual_dim], dim=-1)

        # Determine target network input based on training mode
        concept_preds = self.get_concept_predictions(concept_logits)
        if self.training and self.training_mode == 'independent':
            x = torch.cat([concepts, residual], dim=-1)
        elif self.training and self.training_mode == 'sequential':
            x = torch.cat([concept_preds.detach(), residual], dim=-1)
        else:
            x = torch.cat([concept_preds, residual], dim=-1)

        # Get target logits
        target_logits = self.target_network(x, concepts=concepts)
        return concept_logits, residual, target_logits

    def get_concept_predictions(self, concept_logits: Tensor) -> Tensor:
        """
        Compute concept predictions from logits.
        """
        if self.concept_type == 'binary':
            return concept_logits.sigmoid()

        return concept_logits



### Concept Models with PyTorch Lightning

class ConceptLightningModel(pl.LightningModule):
    """
    Lightning module for training a concept model.
    """

    def __init__(
        self,
        concept_model: ConceptModel,
        concept_loss_fn: Callable = F.binary_cross_entropy_with_logits,
        residual_loss_fn: Callable = lambda r, c: torch.tensor(0.0, device=r.device),
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        concept_loss_fn : Callable(concept_logits, concepts) -> loss
            Concept loss function
        residual_loss_fn : Callable(residual, concept_preds) -> loss
            Residual loss function
        lr : float
            Learning rate
        alpha : float
            Weight for concept loss
        beta : float
            Weight for residual loss
        """
        super().__init__()
        self.concept_model = concept_model
        self.concept_loss_fn = concept_loss_fn
        self.residual_loss_fn = residual_loss_fn
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.log_kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True}

    def dummy_pass(self, loader: Iterable[ConceptBatch]):
        """
        Run a dummy forward pass to handle any uninitialized parameters.

        Parameters
        ----------
        loader : Iterable[ConceptBatch]
            Data loader
        """
        with torch.no_grad():
            (data, concepts), targets = next(iter(loader))
            self.concept_model(data, concepts=concepts)

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass.
        """
        return self.concept_model.forward(*args, **kwargs)

    def loss_fn(
        self,
        batch: ConceptBatch,
        outputs: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Compute loss.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        outputs : tuple[Tensor, Tensor, Tensor]
            Concept model outputs (concept_logits, residual, target_logits)
        """
        (data, concepts), targets = batch
        concept_logits, residual, target_logits = outputs

        if isinstance(concepts, (list, tuple)):
            return oai_loss_fn(self, batch, outputs)

        # Concept loss
        concept_loss = self.concept_loss_fn(concept_logits, concepts)
        if concept_loss.requires_grad:
            self.log('concept_loss', concept_loss, **self.log_kwargs)

        # Residual loss
        concept_preds = self.concept_model.get_concept_predictions(concept_logits)
        residual_loss = self.residual_loss_fn(residual, concept_preds.detach())
        if residual_loss.requires_grad:
            self.log('residual_loss', residual_loss, **self.log_kwargs)

        # Target loss
        target_loss = F.cross_entropy(target_logits, targets)
        if target_loss.requires_grad:
            self.log('target_loss', target_loss, **self.log_kwargs)

        return target_loss + (self.alpha * concept_loss) + (self.beta * residual_loss)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.concept_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def step(
        self,
        batch: ConceptBatch,
        split: Literal['train', 'val', 'test'],
    ) -> Tensor:
        """
        Training / validation / test step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        split : one of {'train', 'val', 'test'}
            Dataset split
        """
        (data, concepts), targets = batch
        outputs = self.concept_model(data, concepts=concepts)
        concept_logits, residual, target_logits = outputs

        # Track loss
        loss = self.loss_fn(batch, outputs)
        self.log(f"{split}_loss", loss, **self.log_kwargs)

        # Track accuracy
        acc = accuracy(target_logits, targets)
        self.log(f"{split}_acc", acc, **self.log_kwargs)

        # Track concept accuracy
        if self.concept_model.concept_type == 'binary':
            concepts = concepts[0] if isinstance(concepts, (list, tuple)) else concepts
            concept_acc = accuracy(concept_logits, concepts, task='binary')
            self.log(f"{split}_concept_acc", concept_acc, **self.log_kwargs)

        return loss

    def training_step(self, batch: ConceptBatch, batch_idx: int) -> Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        return self.step(batch, split='train')

    def validation_step(self, batch: ConceptBatch, batch_idx: int) -> Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        return self.step(batch, split='val')

    def test_step(self, batch: ConceptBatch, batch_idx: int) -> Tensor:
        """
        Test step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        return self.step(batch, split='test')

    def callback(self) -> pl.Callback:
        """
        Callback for this model.
        """
        return pl.Callback()
