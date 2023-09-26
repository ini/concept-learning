from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.cifar import CIFAR100
from datasets.cub import CUB



def get_data_loaders(name='cifar100', batch_size=64):
    if name == 'cifar100':
        concept_dim = 20
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
            transforms.Resize((224, 224)),
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
        concept_dim = 312
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, concept_dim
