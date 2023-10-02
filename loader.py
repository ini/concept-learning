from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.cifar import CIFAR100
from datasets.cub import CUB
from datasets.pitfalls import MNIST_45, DatasetC, DatasetD, DatasetE



def get_data_loaders(
    name: str = 'cifar100',
    batch_size: int = 64) -> tuple[DataLoader, DataLoader, int, int]:
    """
    Get data loaders for the specified dataset.

    Parameters
    ----------
    name : str
        Name of the dataset
    batch_size : int
        Batch size

    Returns
    -------
    train_loader : DataLoader
        Train data loader
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Number of concepts
    num_classes : int
        Number of label classes
    """
    if name == 'pitfalls_mnist_without_45':
        concept_dim, num_classes = 2, 2
        train_dataset, test_dataset = MNIST_45(train=True), MNIST_45(train=False)

    elif name == 'pitfalls_random_concepts':
        concept_dim, num_classes = 100, 2
        train_dataset = DatasetC(num_concepts=100, train=True)
        test_dataset = DatasetC(num_concepts=100, train=False)

    elif name == 'pitfalls_synthetic':
        concept_dim, num_classes = 3, 2
        train_dataset, test_dataset = DatasetD(train=True), DatasetD(train=False)

    elif name == 'pitfalls_mnist_123456':
        concept_dim, num_classes = 3, 2
        train_dataset, test_dataset = DatasetE(train=True), DatasetE(train=False)

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
            root='./data', train=True, transform=transform_train, download=True)
        test_dataset = CIFAR100(
            root='./data', train=False, transform=transform_test, download=True)

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
            root='./data', split='train', transform=transform_train, download=True)
        test_dataset = CUB(
            root='./data', split='test', transform=transform_test, download=True)

    # Make data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, concept_dim, num_classes
