import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .base import ConceptBatch, ConceptModel, ConceptLightningModel
from lib.club import CLUB



class MutualInformationLoss(nn.Module):
    """
    Creates a criterion that estimates an upper bound on the mutual information
    between x and y samples.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64, lr: float = 1e-3):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of x samples
        y_dim : int
            Dimension of y samples
        hidden_dim : int
            Dimension of hidden layers in mutual information estimator
        lr : float
            Learning rate for mutual information estimator optimizer
        """
        super().__init__()
        self.mi_estimator = CLUB(x_dim, y_dim, hidden_dim)
        self.mi_optimizer = torch.optim.Adam(self.mi_estimator.parameters(), lr=lr)

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Estimate (an upper bound on) the mutual information for a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim)
            Batch of x samples
        y : Tensor of shape (..., y_dim)
            Batch of y samples
        """
        return F.softplus(self.mi_estimator.forward(x, y))

    def step(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Run a single training step for the mutual information estimator
        on a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim)
            Batch of x samples
        y : Tensor of shape (..., y_dim)
            Batch of y samples
        """
        # Unfreeze all params for MI estimator training
        self.train()
        for param in self.parameters():
            param.requires_grad = True

        # Train the MI estimator
        self.mi_optimizer.zero_grad()
        estimation_loss = self.mi_estimator.learning_loss(x.detach(), y.detach())
        estimation_loss.backward()
        self.mi_optimizer.step()

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        return estimation_loss


class MutualInfoCallback(nn.Module, pl.Callback):
    """
    Callback class for `MutualInfoConceptLightningModel`.
    """

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: ConceptLightningModel,
        batch: ConceptBatch,
        batch_idx: int):
        """
        Run one training step for the mutual information estimator.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : ConceptLightningModel
            Concept model
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        assert isinstance(pl_module.residual_loss_fn, MutualInformationLoss)

        # Get concepts and residual
        with torch.no_grad():
            (data, concepts), targets = batch
            concept_logits, residual, target_logits = pl_module(data, concepts=concepts)

        # Calculate mutual information estimator loss
        concept_preds = pl_module.concept_model.get_concept_predictions(concept_logits)
        mi_estimator_loss = pl_module.residual_loss_fn.step(residual, concept_preds)
        pl_module.log('mi_estimator_loss', mi_estimator_loss, **pl_module.log_kwargs)


class MutualInfoConceptLightningModel(ConceptLightningModel):
    """
    Concept model that minimizes the mutual information between
    the residual and concept predictions.
    """

    def __init__(
        self,
        concept_model: ConceptModel,
        concept_dim: int,
        residual_dim: int,
        mi_estimator_hidden_dim: int = 64,
        mi_optimizer_lr: float = 1e-3,
        **kwargs):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        concept_dim : int
            Dimension of concept vector
        residual_dim : int
            Dimension of residual vector
        mi_estimator_hidden_dim : int
            Dimension of hidden layer in mutual information estimator
        mi_optimizer_lr : float
            Learning rate for mutual information estimator optimizer
        """
        residual_loss_fn = MutualInformationLoss(
            residual_dim,
            concept_dim,
            hidden_dim=mi_estimator_hidden_dim,
            lr=mi_optimizer_lr,
        )
        super().__init__(concept_model, residual_loss_fn=residual_loss_fn, **kwargs)

    def callback(self) -> MutualInfoCallback:
        """
        Callback for this model.
        """
        return MutualInfoCallback()
