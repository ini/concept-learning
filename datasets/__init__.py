import functools
import torch

from pathlib import Path
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from .cifar import CIFAR100
from .cub import CUB
from .other import MNISTModulo
from .pitfalls import MNIST_45, DatasetC, DatasetD, DatasetE



@functools.cache
def get_datasets(dataset_name: str, data_dir: str) -> tuple[Dataset, Dataset, Dataset]:
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

    if dataset_name == 'mnist_modulo':
        train_dataset = MNISTModulo(root=data_dir, train=True)
        test_dataset = MNISTModulo(root=data_dir, train=False)

    elif dataset_name == 'pitfalls_mnist_without_45':
        train_dataset = MNIST_45(root=data_dir, train=True)
        test_dataset = MNIST_45(root=data_dir, train=False)

    elif dataset_name == 'pitfalls_random_concepts':
        train_dataset = DatasetC(root=data_dir, num_concepts=100, train=True)
        test_dataset = DatasetC(root=data_dir, num_concepts=100, train=False)

    elif dataset_name == 'pitfalls_synthetic':
        train_dataset = DatasetD(train=True)
        val_dataset = DatasetD(train=False)
        test_dataset = DatasetD(train=False)

    elif dataset_name == 'pitfalls_mnist_123456':
        train_dataset = DatasetE(root=data_dir, train=True)
        test_dataset = DatasetE(root=data_dir, train=False)

    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
        ])
        train_dataset = CIFAR100(
            root=data_dir, train=True, transform=transform_train, download=True)
        test_dataset = CIFAR100(
            root=data_dir, train=False, transform=transform_test, download=True)

    elif dataset_name == 'cub':
        transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
        train_dataset = CUB(
            root=data_dir, split='train', transform=transform_train, download=True)
        val_dataset = CUB(
            root=data_dir, split='val', transform=transform_test, download=True)
        test_dataset = CUB(
            root=data_dir, split='test', transform=transform_test, download=True)

    else:
        raise ValueError(f"Invalid dataset name:", dataset_name)

    # Create validation set if necessary
    if val_dataset is None:
        N = len(train_dataset)
        lengths = [N - int(0.15 * N), int(0.15 * N)]
        train_dataset, val_dataset = random_split(
            train_dataset, lengths, generator=torch.Generator().manual_seed(0))

    return train_dataset, val_dataset, test_dataset
