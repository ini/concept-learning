import torch.nn as nn
from .base import ConceptModel



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
        """
        Parameters
        ----------
        concept_network : nn.Module -> (..., concept_dim)
            Concept prediction network
        residual_network : nn.Module -> (..., residual_dim)
            Residual network
        target_network : nn.Module (..., bottleneck_dim) -> (..., num_classes)
            Target network
        concept_dim : int
            Dimension of concept vector
        residual_dim : int
            Dimension of residual vector
        base_network : nn.Module
            Optional base network
        norm_type : str or None
            Type of normalization to apply to bottleneck layer
        T_whitening : int
            Number of iterations to run whitening (if `norm_type` is 'iter_norm')
        affine_whitening : bool
            Whether to apply affine transformation after whitening
        """

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
