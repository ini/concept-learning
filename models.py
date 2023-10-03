from __future__ import annotations

import torch
import torch.nn as nn

from abc import ABC
from lib.iterative_normalization import IterNorm, IterNormRotation
from torch import Tensor



class ConceptModel(nn.Module, ABC):
    """
    Abstract base class for concept models.
    """
    def __init__(self):
        super(ConceptModel, self).__init__()


class ConceptBottleneckModel(ConceptModel):
    """
    Concept bottleneck model.
    """

    def __init__(
        self,
        concept_network: nn.Module,
        residual_network: nn.Module,
        target_network: nn.Module,
        base_network: nn.Module = nn.Identity(),
        config = {},
        **kwargs):
        super().__init__()
        self.config = config
        self.base_network = base_network
        self.concept_network = concept_network
        self.residual_network = residual_network
        self.target_network = target_network
        self.norm_type = config.get("norm_type", "none")
        self.T_whitening = config.get("T_whitening", 0)
        self.combineddim = config["concept_dim"] + config["residual_dim"]
        self.affine_whitening = True
        if self.norm_type == "batch_norm":
            self.ItN = nn.BatchNorm1d(self.combineddim)
        elif self.norm_type == "layer_norm":
            self.ItN = nn.LayerNorm(self.combineddim)
        elif self.norm_type == "instance_norm":
            self.ItN = nn.InstanceNorm1d(self.combineddim, affine=self.affine_whitening)
        elif self.norm_type == "spectral_norm":
            self.ItN = nn.utils.spectral_norm(
                nn.Linear(self.combineddim, self.combineddim)
            )
        elif self.norm_type == "iter_norm" and self.T_whitening > 0:
            self.ItN = IterNorm(
                self.combineddim,
                num_channels=self.combineddim,
                dim=2,
                T=self.T_whitening,
                momentum=1,
                affine=self.affine_whitening,
            )
        else:
            self.ItN = nn.Identity()

    def forward(
        self, x: Tensor,
        concept_preds: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concept_preds : Tensor or None
            Concept values to override the model's concept predictor
            (e.g. for interventions)

        Returns
        -------
        concept_preds : Tensor
            Concept predictions
        residual : Tensor
            Residual vector
        target_preds : Tensor
            Target predictions
        """
        x = self.base_network(x)
        if concept_preds is None:
            concept_preds = self.concept_network(x)

        residual = self.residual_network(x)
        bottleneck_ = torch.cat([concept_preds, residual], dim=-1)
        bottleneck = self.ItN(bottleneck_)
        target_preds = self.target_network(bottleneck)
        return concept_preds, residual, target_preds


class ConceptWhiteningModel(ConceptModel):
    """
    Concept whitening model.
    """

    def __init__(
        self,
        base_network: nn.Module,
        target_network: nn.Module,
        bottleneck_dim: int,
        whitening_activation_mode: str = 'mean',
        **kwargs):
        """
        """
        super().__init__()
        self.base_network = base_network
        self.target_network = target_network
        self.bottleneck_layer = IterNormRotation(
            bottleneck_dim, activation_mode=whitening_activation_mode)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        """
        bottleneck = self.activations(x)
        return self.target_network(bottleneck)

    def activations(self, x: Tensor) -> Tensor:
        """
        Return the output of the concept whitening layer.
        """
        x = self.base_network(x)

        bottleneck = x
        while bottleneck.ndim < 4:
            bottleneck = bottleneck.unsqueeze(-1)

        return self.bottleneck_layer(bottleneck).view(x.shape)
