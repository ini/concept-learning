import pytorch_lightning as pl
import ray
import tempfile
import torch

from pathlib import Path
from ray.train import Checkpoint
from typing import Any

from utils import unwrap



def config_get(config: dict[str, Any], key: str):
    """
    Get a value from a Ray-style configuration dictionary
    (handles top-level grid search).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    key : str
        Configuration key
    """
    if 'grid_search' in config:
        return config['grid_search'][0][key]
    else:
        return config[key]

def config_set(config: dict[str, Any], key: str, value: Any):
    """
    Set a value in a Ray-style configuration dictionary
    (handles top-level grid search).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    key : str
        Configuration key
    value : Any
        Configuration value
    """
    if 'grid_search' in config:
        for config in config['grid_search']:
            config[key] = value
    else:
        config[key] = value


class RayCallback(pl.Callback):
    """
    Callback class for using Ray Tune with PyTorch Lightning.
    """

    def __init__(self, checkpoint_frequency: int = 1, **kwargs):
        """
        Parameters
        ----------
        checkpoint_frequency : int
            Frequency of checkpoints (e.g. every N epochs)
        """
        super().__init__()
        self.checkpoint_frequency = checkpoint_frequency
        self.metrics = {}

    def create_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint_dir: Path | str) -> Checkpoint:
        """
        Create a checkpoint.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : pl.LightningModule
            PyTorch Lightning module
        checkpoint_dir : Path or str
            Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Save current model weights (for backwards compatibility)
        torch.save(
            unwrap(pl_module.concept_model).state_dict(),
            checkpoint_dir / 'model.pt',
        )

        # Save lightning checkpoint
        trainer.save_checkpoint(checkpoint_dir / 'checkpoint.pt')
        return Checkpoint.from_directory(checkpoint_dir)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Clear metrics when a train epoch starts.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : pl.LightningModule
            PyTorch Lightning module
        """
        self.metrics.clear()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Report metrics when a validation epoch ends.
        """
        # Update metrics
        self.metrics.update({k: v.item() for k, v in trainer.callback_metrics.items()})
        self.metrics['epoch'] = self.metrics['step'] = trainer.current_epoch

        # Report metrics
        if ray.train.get_context().get_local_rank() == 0:
            if self.checkpoint_frequency:
                if (trainer.current_epoch + 1) % self.checkpoint_frequency == 0:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        checkpoint = self.create_checkpoint(
                            trainer, pl_module, temp_checkpoint_dir)
                        ray.train.report(metrics=self.metrics, checkpoint=checkpoint)

                else:
                    ray.train.report(metrics=self.metrics)
