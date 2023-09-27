import torch
import torch.nn as nn

from lib.iterative_normalization import IterNormRotation
from torch import Tensor



class ConceptBottleneckModel(nn.Module):
    """
    Concept bottleneck model.
    """

    def __init__(
        self,
        concept_network: nn.Module,
        residual_network: nn.Module,
        target_network: nn.Module,
        base_network: nn.Module = nn.Identity(),
        **kwargs):
        super().__init__()
        self.base_network = base_network
        self.concept_network = concept_network
        self.residual_network = residual_network
        self.target_network = target_network

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
        bottleneck = torch.cat([concept_preds.detach(), residual], dim=-1)
        target_preds = self.target_network(bottleneck)
        return concept_preds, residual, target_preds


class ConceptWhiteningModel(nn.Module):
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
