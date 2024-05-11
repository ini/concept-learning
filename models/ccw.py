from __future__ import annotations
from lib.ccw import EYE

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from typing import Any, Callable, Iterable, Literal

from nn_extensions import VariableKwargs
from utils import accuracy, unwrap, zero_loss_fn
from .base import ConceptModel
from nn_extensions import Chain

### Typing

ConceptBatch = tuple[tuple[Tensor, Tensor], Tensor]  # ((data, concepts), targets)


class CCWConceptLightningModel(pl.LightningModule):
    """
    Lightning module for training a concept model.
    """

    def __init__(
        self,
        concept_model: ConceptModel,
        concept_loss_fn: Callable | None = F.binary_cross_entropy_with_logits,
        residual_loss_fn: Callable | None = None,
        model_type="ccm_eye",
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
        residual_loss_fn : Callable(residual, concepts) -> loss
            Residual loss function
        lr : float
            Learning rate
        alpha : float
            Weight for concept loss
        beta : float
            Weight for residual loss
        """
        if "concept_dim" in kwargs and kwargs["concept_dim"] == 0:
            concept_loss_fn = None
        if "residual_dim" in kwargs and kwargs["residual_dim"] == 0:
            residual_loss_fn = None

        super().__init__()
        self.model_type = model_type
        self.concept_model = concept_model
        self.concept_loss_fn = concept_loss_fn or zero_loss_fn
        self.residual_loss_fn = residual_loss_fn or zero_loss_fn
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": True}

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

        if self.model_type == "ccm_eye":
            # Concept loss
            concept_loss = self.concept_loss_fn(concept_logits, concepts)
            if concept_loss.requires_grad:
                self.log("concept_loss", concept_loss, **self.log_kwargs)

            # Residual loss
            if type(self.concept_model.target_network) == Chain:
                # If the target network is a Chain, the target network is the first module in the chain
                net_y = self.concept_model.target_network[1].module
            else:
                net_y = self.concept_model.target_network.module
            if not isinstance(net_y, nn.Linear):
                net_y = net_y[1]

            device = residual.device
            r = torch.cat(
                [torch.ones(concept_logits.shape[1]), torch.zeros(residual.shape[1])]
            ).to(device)

            residual_loss = EYE(r, net_y.weight.abs().sum(0))
            if residual_loss.requires_grad:
                self.log("residual_loss", residual_loss, **self.log_kwargs)

            # Target loss
            target_loss = F.cross_entropy(target_logits, targets)
            if target_loss.requires_grad:
                self.log("target_loss", target_loss, **self.log_kwargs)

            return (
                target_loss + (self.alpha * concept_loss) + (self.beta * residual_loss)
            )
        else:
            raise ValueError("Unknown model type:", self.model_type)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.concept_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def step(
        self,
        batch: ConceptBatch,
        split: Literal["train", "val", "test"],
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
        if self.concept_model.concept_type == "binary":
            concept_acc = accuracy(concept_logits, concepts, task="binary")
            self.log(f"{split}_concept_acc", concept_acc, **self.log_kwargs)
        else:
            concept_rmse = F.mse_loss(concept_logits, concepts).sqrt()
            self.log(f"{split}_concept_rmse", concept_rmse, **self.log_kwargs)

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
        return self.step(batch, split="train")

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
        return self.step(batch, split="val")

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
        return self.step(batch, split="test")
