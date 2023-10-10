from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

from lib.iterative_normalization import IterNorm, IterNormRotation
from nn_extensions import Apply
from utils import unwrap



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


class ConceptWhitening(IterNormRotation):
    """
    Concept whitening layer (with support for arbitrary number of input dimensions).
    """
    def forward(self, x: Tensor):
        bottleneck = x
        while bottleneck.ndim < 4:
            bottleneck = bottleneck.unsqueeze(-1)
        return super().forward(bottleneck).view(*x.shape)



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
        generates concept predictions
    residual_network : nn.Module
        Network that takes the output of `base_network` and
        generates a residual vector
    bottleneck_layer : nn.Module
        Network that post-processes the concatenated output of
        `concept_network` and `residual_network`
    target_network : nn.Module
        Network that takes the output of `bottleneck_layer` and
        generates target predictions
    """

    def __init__(
        self,
        concept_network: nn.Module,
        residual_network: nn.Module,
        target_network: nn.Module,
        base_network: nn.Module = nn.Identity(),
        bottleneck_layer: nn.Module = nn.Identity()):
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
        """
        super().__init__()
        self.base_network = ConceptModuleWrapper(base_network)
        self.concept_network = ConceptModuleWrapper(concept_network)
        self.residual_network = ConceptModuleWrapper(residual_network)
        self.bottleneck_layer = ConceptModuleWrapper(bottleneck_layer)
        self.target_network = ConceptModuleWrapper(target_network)

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
        # Get concept predictions & residual
        x = self.base_network(x, concepts=concepts)
        concept_preds = self.concept_network(x, concepts=concepts)
        residual = self.residual_network(x, concepts=concepts)

        # Process concept predictions & residual via bottleneck layer
        if not isinstance(unwrap(self.bottleneck_layer), nn.Identity):
            x = torch.cat([concept_preds, residual], dim=-1)
            x = self.bottleneck_layer(x, concepts=concepts)
            concept_preds, residual = x.split(
                [concept_preds.shape[-1], residual.shape[-1]], dim=-1)

        # Get target predictions
        bottleneck = torch.cat([concept_preds.detach(), residual], dim=-1)
        target_preds = self.target_network(bottleneck, concepts=concepts)

        return concept_preds, residual, target_preds


class ConceptBottleneckModel(ConceptModel):
    """
    Concept bottleneck model.
    """

    def __init__(
        self,
        concept_network: nn.Module,
        residual_network: nn.Module,
        target_network: nn.Module,
        concept_dim: int,
        residual_dim: int,
        base_network: nn.Module = nn.Identity(),
        norm_type: str | None = None,
        T_whitening: int = 0,
        affine_whitening: bool = True,
        **kwargs):

        bottleneck_dim = concept_dim + residual_dim
        if norm_type == 'batch_norm':
            bottleneck_layer = nn.BatchNorm1d(bottleneck_dim)
        elif norm_type == 'layer_norm':
            bottleneck_layer = nn.LayerNorm(bottleneck_dim)
        elif norm_type == 'instance_norm':
            bottleneck_layer = nn.InstanceNorm1d(bottleneck_dim, affine=affine_whitening)
        elif norm_type == 'spectral_norm':
            bottleneck_layer = nn.utils.spectral_norm(
                nn.Linear(bottleneck_dim, bottleneck_dim))
        elif norm_type == 'iter_norm' and T_whitening > 0:
            bottleneck_layer = IterNorm(
                bottleneck_dim,
                num_channels=bottleneck_dim,
                T=T_whitening,
                dim=2,
                momentum=1,
                affine=affine_whitening,
            )
        else:
            bottleneck_layer = nn.Identity()

        super().__init__(
            concept_network=concept_network,
            residual_network=residual_network,
            target_network=target_network,
            base_network=base_network,
            bottleneck_layer=bottleneck_layer,
        )


class ConceptWhiteningModel(ConceptModel):
    """
    Concept whitening model.
    """

    def __init__(
        self,
        base_network: nn.Module,
        target_network: nn.Module,
        concept_dim: int,
        residual_dim: int,
        T_whitening: int = 0,
        affine_whitening: bool = True,
        activation_mode: str = 'mean',
        **kwargs):

        bottleneck_layer = ConceptWhitening(
            concept_dim + residual_dim,
            num_channels=concept_dim + residual_dim,
            T=T_whitening,
            dim=4,
            momentum=1.0,
            affine=affine_whitening,
            activation_mode=activation_mode,
        )

        super().__init__(
            concept_network=Apply(lambda x: x[..., :concept_dim]),
            residual_network=Apply(lambda x: x[..., -residual_dim:]),
            target_network=target_network,
            base_network=base_network,
            bottleneck_layer=bottleneck_layer,
        )
