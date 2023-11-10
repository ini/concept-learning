import functools
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Any
from datasets import get_datasets



DATASET_INFO = {
    'mnist_modulo': {'concept_dim': 5, 'num_classes': 10},
    'pitfalls_mnist_without_45': {'concept_dim': 2, 'num_classes': 2},
    'pitfalls_random_concepts': {'concept_dim': 100, 'num_classes': 2},
    'pitfalls_synthetic': {'concept_dim': 3, 'num_classes': 2},
    'pitfalls_mnist_123456': {'concept_dim': 3, 'num_classes': 2},
    'cifar100': {'concept_dim': 20, 'num_classes': 100},
    'cub': {'concept_dim': 112, 'num_classes': 200},
    'oai': {'concept_type': 'continuous', 'concept_dim': 10, 'num_classes': 4},
}



def get_datamodule(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> pl.LightningDataModule:
    """
    Get a LightningDataModule for the specified dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)
    batch_size : int
        Batch size
    num_workers : int
        Number of workers for the data loaders
    """
    train_dataset, val_dataset, test_dataset = get_datasets(dataset_name, data_dir)
    return pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers, # TODO: fix Ray Tune pickling error when num_workers > 0
    )

@functools.cache
def get_dummy_batch(dataset_name: str, data_dir: str) -> tuple[Any, Any]:
    """
    Get dummy batch for the specified dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)
    """
    loader = get_datamodule(dataset_name, data_dir).train_dataloader()
    return next(iter(loader))

@functools.cache
def get_concept_loss_fn(dataset_name: str, data_dir: str) -> nn.BCEWithLogitsLoss:
    """
    Get BCE concept loss function for the specified dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)
    """
    if dataset_name == 'oai':
        # Get weighted mean squared error loss
        def weighted_mse(input, target):
            loss = (input - target) ** 2
            loss *= target.not_nan
            loss *= target.loss_class_wts
            return loss.mean()

        return weighted_mse

    else:
        # Get weighted binary cross entropy loss
        train_loader = get_datamodule(dataset_name, data_dir).train_dataloader()
        concept_dim = DATASET_INFO[dataset_name]['concept_dim']
        concepts_pos_count = torch.zeros(concept_dim)
        concepts_neg_count = torch.zeros(concept_dim)
        for (data, concepts), targets in train_loader:
            concepts_pos_count += concepts.sum(dim=0)
            concepts_neg_count += (1 - concepts).sum(dim=0)

        pos_weight = concepts_neg_count / (concepts_pos_count + 1e-6)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
