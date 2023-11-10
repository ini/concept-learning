import functools
import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from .cifar import CIFAR100
from .cub import CUB
from .oai import OAI
from .other import MNISTModulo
from .pitfalls import MNIST_45, DatasetC, DatasetD, DatasetE



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

    elif dataset_name == 'oai':
        transform_train = transforms.Compose([
            transforms.Resize(224, antialias=False) if resize_oai else lambda x: x,
            transforms.Normalize(mean=-31334.48612, std=1324.959356),
            RandomTranslation(0.1, 0.1),
            lambda x: x.expand(3, -1, -1) # expand to 3 channels
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, antialias=False) if resize_oai else lambda x: x,
            transforms.Normalize(mean=-31334.48612, std=1324.959356),
            lambda x: x.expand(3, -1, -1) # expand to 3 channels
        ])
        train_dataset = OAI(
            root=data_dir, split='train', transform=transform_train)
        val_dataset = OAI(
            root=data_dir, split='val', transform=transform_test)
        test_dataset = OAI(
            root=data_dir, split='test', transform=transform_test)

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
