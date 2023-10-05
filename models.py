from __future__ import annotations

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from lib.iterative_normalization import IterNorm, IterNormRotation
from torch import Tensor



class ArgsWrapper(nn.Module):
    """
    Wrapper to allow model to take additional arguments.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        if name == 'model':
            return super().__getattr__(name)
        else:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        return self.model.forward(args[0])


class ConceptModel(nn.Module, ABC):
    """
    Abstract base class for concept models.
    """
    def __init__(self):
        super(ConceptModel, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor, concepts: Tensor | None = None):
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values
        """
        pass


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
        self.concept_network = ArgsWrapper(concept_network)
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
        target_preds : Tensor
            Target predictions
        """
        x = self.base_network(x)
        concept_preds = self.concept_network(x, concepts=concepts)
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

        class CW(IterNormRotation):
            def forward(self, x):
                bottleneck = x
                while bottleneck.ndim < 4:
                    bottleneck = bottleneck.unsqueeze(-1)
                return super().forward(bottleneck).view(*x.shape)

        self.bottleneck_layer = CW(
            bottleneck_dim, activation_mode=whitening_activation_mode)
        self.bottleneck_layer = ArgsWrapper(self.bottleneck_layer)

    def forward(self, x: Tensor, concepts: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values
        """
        bottleneck = self.activations(x, concepts=concepts)
        return self.target_network(bottleneck)

    def activations(self, x: Tensor, concepts: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values
        """
        x = self.base_network(x)
        return self.bottleneck_layer(x, concepts=concepts)
