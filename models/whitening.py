import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch import Tensor

from .base import ConceptBatch, ConceptModel, ConceptLightningModel
from .bottleneck import ConceptWhitening
from utils import unwrap



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
        assert isinstance(
            unwrap(pl_module.concept_model.bottleneck_layer), ConceptWhitening)

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
        cw_alignment_frequency: int = 20,
        **kwargs):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        cw_alignment_frequency : int
            Frequency of concept alignment (e.g. every N batches)
        """
        assert isinstance(unwrap(concept_model.bottleneck_layer), ConceptWhitening)
        self._callback = ConceptWhiteningCallback(
            alignment_frequency=cw_alignment_frequency)
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
        if target_loss.requires_grad:
            self.log('target_loss', target_loss, **self.log_kwargs)

        return target_loss

    def callback(self) -> ConceptWhiteningCallback:
        """
        Callback for this model.
        """
        return self._callback
