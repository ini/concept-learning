import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Literal

from .base import ConceptBatch, ConceptModel, ConceptLightningModel
from lib.iterative_normalization import IterNormRotation
from nn_extensions import Apply



class ConceptWhitening(IterNormRotation):
    """
    Concept whitening layer (with support for arbitrary number of input dimensions).
    """

    def forward(self, x: Tensor):
        bottleneck = x
        while bottleneck.ndim < 4:
            bottleneck = bottleneck.unsqueeze(-1)

        return super().forward(bottleneck).view(*x.shape)


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
        activation_mode: str = Literal['mean', 'max', 'pos_mean', 'pool_max'],
        **kwargs):
        """
        Parameters
        ----------
        base_network : nn.Module -> (..., bottleneck_dim)
            Base network
        target_network : nn.Module (..., bottleneck_dim) -> (..., num_classes)
            Target network 
        concept_dim : int
            Dimension of concept vector
        residual_dim : int
            Dimension of residual vector
        T_whitening : int
            Number of iterations for whitening layer
        affine_whitening : bool
            Whether to use affine whitening
        activation_mode : one of {'mean', 'max', 'pos_mean', 'pool_max'}
            Mode for concept whitening activation
        """
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
            **dict(kwargs, training_mode='joint'),
        )


class ConceptWhiteningCallback(pl.Callback):
    """
    Callback class for `ConceptWhiteningLightningModel`.
    """

    def __init__(self, alignment_frequency: int = 20):
        """
        Parameters
        ----------
        alignment_frequency : int
            Frequency of concept alignment (e.g. every N batches)
        """
        super().__init__()
        self.concept_loaders = None
        self.alignment_frequency = alignment_frequency

    def on_train_start(self, trainer: pl.Trainer, pl_module: ConceptLightningModel):
        """
        Create concept data loaders.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : ConceptLightningModel
            Concept model
        """
        assert isinstance(pl_module.concept_model, ConceptWhiteningModel)

        # Get training data loader
        loader = trainer.fit_loop._data_source.instance
        (data, concepts), targets = next(iter(loader))
        batch_size, concept_dim = concepts.shape

        # Create concept data loaders (one for each concept)
        self.concept_loaders = []
        try:
            for concept_idx in range(concept_dim):
                concept_loader = torch.utils.data.DataLoader(
                    dataset=[x for ((x, c), y) in loader.dataset if c[concept_idx] == 1],
                    batch_size=batch_size,
                    shuffle=True,
                )
                self.concept_loaders.append(concept_loader)
        except ValueError as e:
            print('Error creating concept loaders:', e)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: ConceptLightningModel,
        batch: ConceptBatch,
        batch_idx: int):
        """
        Align concepts in the concept whitening layer.

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
        if self.concept_loaders is None:
            return

        if (batch_idx + 1) % self.alignment_frequency == 0:
            pl_module.freeze()
            with torch.no_grad():
                for concept_idx, concept_loader in enumerate(self.concept_loaders):
                    pl_module.concept_model.bottleneck_layer.mode = concept_idx
                    for X in concept_loader:
                        X = X.requires_grad_().to(pl_module.device)
                        pl_module.concept_model(X)
                        break

                    pl_module.concept_model.bottleneck_layer.update_rotation_matrix()
                    pl_module.concept_model.bottleneck_layer.mode = -1

            pl_module.unfreeze()


class ConceptWhiteningLightningModel(ConceptLightningModel):
    """
    Concept model that uses concept whitening to decorrelate, normalize,
    and align the latent space with concepts.
    """

    def __init__(
        self,
        concept_model: ConceptModel,
        whitening_alignment_frequency: int = 20,
        **kwargs):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        whitening_alignment_frequency : int
            Frequency of concept alignment (e.g. every N batches)
        """
        self._callback = ConceptWhiteningCallback(
            alignment_frequency=whitening_alignment_frequency)
        super().__init__(concept_model, **kwargs)

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
        target_loss = F.cross_entropy(target_logits, targets)
        self.log('target_loss', target_loss, **self.log_kwargs)

        return target_loss

    def callback(self) -> ConceptWhiteningCallback:
        """
        Callback for this model.
        """
        return self._callback
