from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Accuracy
from typing import Any, Callable, Iterable, Literal

from utils import unwrap



### Typing

ConceptBatch = tuple[tuple[Tensor, Tensor], Tensor] # ((data, concepts), targets)



### Helper Modules

class ConceptModuleWrapper(nn.Module):
    """
    Wrapper to allow module to optionally take in ground truth concepts.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def __getattr__(self, name: str):
        if name == 'module':
            return super().__getattr__(name)
        else:
            return getattr(self.module, name)

    def forward(self, x: Tensor, concepts: Tensor | None = None):
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values
        """
        return self.module(x)



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
        training_mode: Literal['independent', 'sequential', 'joint'] = 'independent',
        **kwargs):
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
        concept_activation : nn.Module
            Concept activation function
        training_mode : one of {'independent', 'sequential', 'joint'}
            Training mode (see https://arxiv.org/abs/2007.04612)
        """
        super().__init__()
        self.base_network = ConceptModuleWrapper(base_network)
        self.concept_network = ConceptModuleWrapper(concept_network)
        self.residual_network = ConceptModuleWrapper(residual_network)
        self.bottleneck_layer = ConceptModuleWrapper(bottleneck_layer)
        self.target_network = ConceptModuleWrapper(target_network)
        self.training_mode = training_mode

    def forward(
        self,
        x: Tensor,
        concepts: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values

        Returns
        -------
        concept_preds : Tensor
            Concept predictions
        residual : Tensor
            Residual vector
        target_logits : Tensor
            Target logits
        """
        # Get concept logits & residual
        x = self.base_network(x, concepts=concepts)
        concept_logits = self.concept_network(x, concepts=concepts)
        residual = self.residual_network(x, concepts=concepts)

        # Process concept logits & residual via bottleneck layer
        if not isinstance(unwrap(self.bottleneck_layer), nn.Identity):
            x = torch.cat([concept_logits, residual], dim=-1)
            x = self.bottleneck_layer(x, concepts=concepts)
            concept_logits, residual = x.split(
                [concept_logits.shape[-1], residual.shape[-1]], dim=-1)

        # Determine target network input based on training mode
        concept_preds = concept_logits.sigmoid()
        if self.training and self.training_mode == 'independent':
            x = torch.cat([concepts, residual], dim=-1)
        elif self.training and self.training_mode == 'sequential':
            x = torch.cat([concept_preds.detach(), residual], dim=-1)
        else:
            x = torch.cat([concept_preds, residual], dim=-1)

        # Get target logits
        target_logits = self.target_network(x, concepts=concepts)
        return concept_preds, residual, target_logits



### Concept Models with PyTorch Lightning

class ConceptLightningModel(pl.LightningModule):
    """
    Lightning module for training a concept model.
    """

    def __init__(
        self,
        concept_model: ConceptModel,
        residual_loss_fn: Callable = lambda r, c: torch.tensor(0.0, device=r.device),
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        residual_loss_fn : Callable(residual, concepts) -> loss
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
        self, batch: ConceptBatch, outputs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        Compute loss.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        outputs : tuple[Tensor, Tensor, Tensor]
            Concept model outputs (concept_preds, residual, target_logits)
        """
        (data, concepts), targets = batch
        concept_preds, residual, target_logits = outputs

        # Concept loss
        concepts = concepts.view(concept_preds.shape)
        concept_loss = F.binary_cross_entropy(concept_preds, concepts)
        self.log('concept_loss', concept_loss, **self.log_kwargs)

        # Residual loss
        residual_loss = self.residual_loss_fn(residual, concept_preds)
        self.log('residual_loss', residual_loss, **self.log_kwargs)

        # Target loss
        target_loss = F.cross_entropy(target_logits, targets)
        self.log('target_loss', target_loss, **self.log_kwargs)

        return target_loss + (self.alpha * concept_loss) + (self.beta * residual_loss)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

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
        (data, concepts), targets = batch
        outputs = self.concept_model(data, concepts=concepts)
        loss = self.loss_fn(batch, outputs)
        self.log('loss', loss, **self.log_kwargs)
        return loss

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
        (data, concepts), targets = batch
        outputs = self.concept_model(data, concepts=concepts)
        concept_preds, residual, target_logits = outputs

        # Track validation loss
        loss = self.loss_fn(batch, outputs)
        self.log('val_loss', loss, **self.log_kwargs)

        # Track validation accuracy
        target_preds = target_logits.argmax(dim=-1)
        accuracy_fn = Accuracy(
            task='multiclass', num_classes=target_logits.shape[-1]).to(self.device)
        accuracy = accuracy_fn(target_preds, targets)
        self.log('val_acc', accuracy, **self.log_kwargs)

        # Track validation concept accuracy
        concept_accuracy_fn = Accuracy(task='binary').to(self.device)
        concept_accuracy = concept_accuracy_fn(concept_preds, concepts)
        self.log('val_concept_acc', concept_accuracy, **self.log_kwargs)

        return loss

    def callback(self) -> pl.Callback:
        """
        Callback for this model.
        """
        return pl.Callback()
