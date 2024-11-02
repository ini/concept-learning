import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any

from .base import ConceptModel, ConceptLightningModel
from lib.club import CLUB


class ConceptResidualConceptPred(nn.Module):
    """ """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        binary: bool = True,
    ):
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
        self.fc = nn.Linear(x_dim, y_dim)
        self.mi_optimizer = torch.optim.RMSprop(self.fc.parameters(), lr=lr)
        self.binary = binary

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Estimate (an upper bound on) the mutual information for a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim) #residual
            Batch of x samples
        """
        y_pred = self.fc(x)
        # if self.binary:
        #     y_pred = torch.sigmoid(y_pred)
        return y_pred

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
        # Forward pass
        y_pred = self.forward(x)

        # Compute the loss
        if self.binary:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y)
        loss.backward()
        self.mi_optimizer.step()

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        return loss


class ConceptEmbeddingConceptPred(nn.Module):
    """ """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        binary: bool = True,
    ):
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
        concept_prob_generators = torch.nn.ModuleList()
        for i in range(y_dim):
            concept_prob_generators.append(
                torch.nn.Linear(
                    (y_dim - 1) * x_dim,
                    1,
                )
            )
        self.concept_prob_generators = concept_prob_generators
        self.mi_optimizer = torch.optim.RMSprop(
            self.concept_prob_generators.parameters(), lr=lr
        )
        self.binary = binary

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Estimate (an upper bound on) the mutual information for a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim) #residual
            Batch of x samples
        """
        batch_size, num_concepts, embed_dim = x.shape
        probs = []
        for i in range(num_concepts):
            # Exclude the i-th concept
            x_except_i = torch.cat([x[:, :i, :], x[:, i + 1 :, :]], dim=1)
            # Reshape to (batch_size, -1)
            print(x_except_i.shape)
            x_except_i = x_except_i.view(batch_size, -1)
            print(x_except_i.shape)
            prob = self.concept_prob_generators[i](x_except_i)
            probs.append(prob)
        y_pred = torch.stack(probs, dim=1).squeeze()
        return y_pred

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
        # Forward pass
        y_pred = self.forward(x)

        # Compute the loss
        if self.binary:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y)
        loss.backward()
        self.mi_optimizer.step()

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        return loss