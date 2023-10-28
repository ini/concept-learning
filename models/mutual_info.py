import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor

from .base import ConceptBatch, ConceptModel, ConceptLightningModel
from lib.club import CLUB



class MutualInfoCallback(nn.Module, pl.Callback):
    """
    Callback class for `MutualInfoConceptLightningModel`.
    """

    def __init__(
        self,
        concept_dim: int,
        residual_dim: int,
        mi_estimator_hidden_dim: int = 64,
        mi_optimizer_lr: float = 1e-3):
        """
        Parameters
        ----------
        concept_dim : int
            Dimension of concept vector
        residual_dim : int
            Dimension of residual vector
        mi_estimator_hidden_dim : int
            Dimension of hidden layer in mutual information estimator
        mi_optimizer_lr : float
            Learning rate for mutual information estimator optimizer
        """
        super().__init__()
        self.mi_estimator = CLUB(residual_dim, concept_dim, mi_estimator_hidden_dim)
        self.mi_optimizer = torch.optim.Adam(
            self.mi_estimator.parameters(), lr=mi_optimizer_lr)

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
        # Run inference for concept model
        with torch.no_grad():
            (data, concepts), targets = batch
            concept_logits, residual, target_logits = pl_module(data, concepts=concepts)

        # Optimize the MI estimator
        self.mi_optimizer.zero_grad()
        concept_preds = pl_module.concept_model.get_concept_predictions(concept_logits)
        mi_loss = self.mi_estimator.learning_loss(residual, concept_preds)
        mi_loss.backward()
        self.mi_optimizer.step()
        self.log('mi_loss', mi_loss, **pl_module.log_kwargs)


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
        super().__init__(
            concept_model,
            residual_loss_fn=self.mutual_information,
            **kwargs,
        )
        self._callback = MutualInfoCallback(
            concept_dim,
            residual_dim,
            mi_estimator_hidden_dim=mi_estimator_hidden_dim,
            mi_optimizer_lr=mi_optimizer_lr,
        )

    def mutual_information(self, residual: Tensor, concept_logits: Tensor) -> Tensor:
        """
        Estimated mutual information between residual and concept predictions.
        """
        concept_preds = self.concept_model.get_concept_predictions(concept_logits)
        return self._callback.mi_estimator(residual, concept_preds.detach())

    def callback(self) -> MutualInfoCallback:
        """
        Callback for this model.
        """
        return self._callback
