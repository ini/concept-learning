import torch

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from datasets.cifar import CIFAR100
from datasets.cub import CUB
from datasets.pitfalls import MNIST_45, DatasetC, DatasetD, DatasetE



DATASET_NAMES = [
    'pitfalls_mnist_without_45', 'pitfalls_random_concepts',
    'pitfalls_synthetic', 'pitfalls_mnist_123456',
    'cifar100', 'cub',
]

def get_data_loaders(
    name: str = 'cifar100',
    data_dir: str = './data',
    batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Get data loaders for the specified dataset.

    Parameters
    ----------
    name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)
    batch_size : int
        Batch size

    Returns
    -------
    train_loader : DataLoader
        Train data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Number of concepts
    num_classes : int
        Number of label classes
    """
    train_dataset, val_dataset, test_dataset = None, None, None

    if name == 'pitfalls_mnist_without_45':
        concept_dim, num_classes = 2, 2
        train_dataset = MNIST_45(root=data_dir, train=True)
        test_dataset = MNIST_45(root=data_dir, train=False)

    elif name == 'pitfalls_random_concepts':
        concept_dim, num_classes = 100, 2
        train_dataset = DatasetC(root=data_dir, num_concepts=100, train=True)
        test_dataset = DatasetC(root=data_dir, num_concepts=100, train=False)

    elif name == 'pitfalls_synthetic':
        concept_dim, num_classes = 3, 2
        train_dataset = DatasetD(train=True)
        val_dataset = DatasetD(train=False)
        test_dataset = DatasetD(train=False)

    elif name == 'pitfalls_mnist_123456':
        concept_dim, num_classes = 3, 2
        train_dataset = DatasetE(root=data_dir, train=True)
        test_dataset = DatasetE(root=data_dir, train=False)

    elif name == 'cifar100':
        concept_dim, num_classes = 20, 100
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

    elif name == 'cub':
        concept_dim, num_classes = 312, 200
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

    # Get validation set
    if val_dataset is None:
        N = len(train_dataset)
        train_dataset, val_dataset = random_split(
            train_dataset,
            [N - int(0.15 * N), int(0.15 * N)],
            generator=torch.Generator().manual_seed(42),
        )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, concept_dim, num_classes
