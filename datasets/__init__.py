import functools
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from typing import Any

from .cifar import CIFAR100
from .cub import CUB
from .oai import OAI
from .other import MNISTModulo
from .pitfalls import MNIST_45, DatasetC, DatasetD, DatasetE
from .imagenet import ImageNet
from .celeba import generate_data as celeba_generate_data
from .aa2 import AA2

DATASET_INFO = {
    "mnist_modulo": {"concept_type": "binary", "concept_dim": 5, "num_classes": 10},
    "pitfalls_mnist_without_45": {
        "concept_type": "binary",
        "concept_dim": 2,
        "num_classes": 2,
    },
    "pitfalls_random_concepts": {
        "concept_type": "binary",
        "concept_dim": 100,
        "num_classes": 2,
    },
    "pitfalls_synthetic": {
        "concept_type": "binary",
        "concept_dim": 3,
        "num_classes": 2,
    },
    "pitfalls_mnist_123456": {
        "concept_type": "binary",
        "concept_dim": 3,
        "num_classes": 2,
    },
    "cifar100": {"concept_type": "binary", "concept_dim": 20, "num_classes": 100},
    "cub": {"concept_type": "binary", "concept_dim": 112, "num_classes": 200},
    "oai": {"concept_type": "continuous", "concept_dim": 10, "num_classes": 4},
    "imagenet": {"concept_type": "binary", "concept_dim": 65, "num_classes": 1000},
    "celeba": {"concept_type": "binary", "concept_dim": 6, "num_classes": 256},
    "aa2": {"concept_type": "binary", "concept_dim": 85, "num_classes": 50},
}


@functools.cache
def get_datasets(
    dataset_name: str,
    data_dir: str,
    resize_oai: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Get train, validation, and test splits for the given dataset.

    Parameters
    ----------
    name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)

    Returns
    -------
    train_dataset : Dataset
        Train dataset
    val_dataset : Dataset
        Validation dataset
    test_dataset : Dataset
        Test dataset
    """
    train_dataset, val_dataset, test_dataset = None, None, None

    if dataset_name == "mnist_modulo":
        train_dataset = MNISTModulo(root=data_dir, train=True)
        test_dataset = MNISTModulo(root=data_dir, train=False)

    elif dataset_name == "pitfalls_mnist_without_45":
        train_dataset = MNIST_45(root=data_dir, train=True)
        test_dataset = MNIST_45(root=data_dir, train=False)

    elif dataset_name == "pitfalls_random_concepts":
        train_dataset = DatasetC(root=data_dir, num_concepts=100, train=True)
        test_dataset = DatasetC(root=data_dir, num_concepts=100, train=False)

    elif dataset_name == "pitfalls_synthetic":
        train_dataset = DatasetD(train=True)
        val_dataset = DatasetD(train=False)
        test_dataset = DatasetD(train=False)

    elif dataset_name == "pitfalls_mnist_123456":
        train_dataset = DatasetE(root=data_dir, train=True)
        test_dataset = DatasetE(root=data_dir, train=False)

    elif dataset_name == "cifar100":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
            ]
        )
        train_dataset = CIFAR100(
            root=data_dir, train=True, transform=transform_train, download=True
        )
        test_dataset = CIFAR100(
            root=data_dir, train=False, transform=transform_test, download=True
        )

    elif dataset_name == "cub":
        transform_train = transforms.Compose(
            [
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )
        train_dataset = CUB(
            root=data_dir, split="train", transform=transform_train, download=True
        )
        val_dataset = CUB(
            root=data_dir, split="val", transform=transform_test, download=True
        )
        test_dataset = CUB(
            root=data_dir, split="test", transform=transform_test, download=True
        )

    elif dataset_name == "oai":
        transform_train = transforms.Compose(
            [
                transforms.Resize(224, antialias=False) if resize_oai else lambda x: x,
                transforms.Normalize(mean=-31334.48612, std=1324.959356),
                RandomTranslation(0.1, 0.1),
                lambda x: x.expand(3, -1, -1),  # expand to 3 channels
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(224, antialias=False) if resize_oai else lambda x: x,
                transforms.Normalize(mean=-31334.48612, std=1324.959356),
                lambda x: x.expand(3, -1, -1),  # expand to 3 channels
            ]
        )
        train_dataset = OAI(root=data_dir, split="train", transform=transform_train)
        val_dataset = OAI(root=data_dir, split="val", transform=transform_test)
        test_dataset = OAI(root=data_dir, split="test", transform=transform_test)
    elif dataset_name == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = ImageNet(
            data_dir + "train/",
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        test_dataset = ImageNet(
            data_dir + "val/",
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

    elif dataset_name == "celeba":
        dataset_config = {
            "dataset": "celeba",
            "root_dir": data_dir,
            "image_size": 64,
            "num_classes": 1000,
            "use_imbalance": True,
            "use_binary_vector_class": True,
            "num_concepts": 6,
            "label_binary_width": 1,
            "label_dataset_subsample": 12,
            "num_hidden_concepts": 2,
            "selected_concepts": False,
            "num_workers": 8,
            "sampling_percent": 1,
            "test_subsampling": 1,
        }
        train_dataset, test_dataset, val_dataset = celeba_generate_data(
            dataset_config, dataset_config["root_dir"], split="all"
        )

    elif dataset_name == "aa2":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = AA2(
            root=data_dir,
            split="train",
            transform=transform_train,
            download=True,
        )
        test_dataset = AA2(
            root=data_dir,
            split="val",
            transform=transform_test,
            download=True,
        )

    else:
        raise ValueError(f"Invalid dataset name:", dataset_name)

    # Create validation set if necessary
    if val_dataset is None:
        N = len(train_dataset)
        train_dataset, val_dataset = random_split(
            train_dataset,
            lengths=[N - int(0.15 * N), int(0.15 * N)],
            generator=torch.Generator().manual_seed(0),
        )

    return train_dataset, val_dataset, test_dataset


def get_datamodule(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    resize_oai: bool = True,
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
    train_dataset, val_dataset, test_dataset = get_datasets(
        dataset_name, data_dir, resize_oai
    )
    return pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # TODO: fix Ray Tune pickling error when num_workers > 0
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
    if dataset_name == "oai":
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
        concept_dim = DATASET_INFO[dataset_name]["concept_dim"]
        concepts_pos_count = torch.zeros(concept_dim)
        concepts_neg_count = torch.zeros(concept_dim)
        for (data, concepts), targets in train_loader:
            concepts_pos_count += concepts.sum(dim=0)
            concepts_neg_count += (1 - concepts).sum(dim=0)

        pos_weight = concepts_neg_count / (concepts_pos_count + 1e-6)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class RandomTranslation:
    """
    Random translation transform.
    """

    def __init__(self, max_dx: float, max_dy: float, seed: int = 0):
        """
        Parameters
        ----------
        max_dx : float in interval [0, 1]
            Maximum absolute fraction for horizontal translation
        max_dy : float in interval [0, 1]
            Maximum absolute fraction for vertical translation
        seed : int
            Seed for the random number generator
        """
        self.max_dx, self.max_dy = max_dx, max_dy
        self.random = np.random.default_rng(seed)

    def __call__(self, img: Tensor) -> Tensor:
        dx = int(self.max_dx * img.shape[-2] * self.random.uniform(-1, -1))
        dy = int(self.max_dy * img.shape[-1] * self.random.uniform(-1, -1))
        return torch.roll(img, shifts=(dx, dy), dims=(-2, -1))
