import torch

from torch.utils.data import Dataset
from torchvision import datasets, transforms



class MNIST_45(Dataset):
    """
    MNIST dataset filtered to exclude digits 4 and 5,
    where the task is to predict the parity of the digit.

    See "Promises and Pitfalls of Black-Box Concept Learning Models"
    (https://arxiv.org/pdf/2106.13314.pdf).
    """

    def __init__(self, train: bool = True):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        mnist = datasets.MNIST(
            root='./data', train=train, transform=transform, download=True)

        # Filter all digits except 4 and 5
        idx = torch.cat([
            torch.argwhere(mnist.targets == digit)[:, 0]
            for digit in set(range(10)) - {4, 5}
        ], dim=0)
        self.data = mnist.data[idx].float()
        self.targets = mnist.targets[idx].long() % 2
        self.concepts = torch.zeros(len(self.data), 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return (self.data[idx], self.concepts[idx]), self.targets[idx]


class DatasetC(Dataset):
    """
    See "Promises and Pitfalls of Black-Box Concept Learning Models"
    Appendix C (https://arxiv.org/pdf/2106.13314.pdf).
    """
    
    def __init__(self, num_concepts: int, train: bool = True):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        mnist = datasets.MNIST(
            root='./data', train=train, transform=transform, download=True)

        # Filter digits (0, 1, 6, 7) and select 500 samples per digit
        idx = torch.cat([
            torch.argwhere(mnist.targets == digit)[:500, 0]
            for digit in (0, 1, 6, 7)
        ], dim=0)
        self.data = mnist.data[idx].float()
        self.targets = mnist.targets[idx].long() % 2

        # Generate random concepts
        X = self.data.view(self.data.shape[0], -1)
        A = torch.rand(X.shape[1], num_concepts)
        S = X @ A
        S_min, S_max = S.min(dim=0).values, S.max(dim=0).values
        B = torch.rand(num_concepts) * (S_max - S_min) + S_min
        self.concepts = (S < B).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return (self.data[idx], self.concepts[idx]), self.targets[idx]


class DatasetD(Dataset):
    """
    See "Promises and Pitfalls of Black-Box Concept Learning Models"
    Appendix D (https://arxiv.org/pdf/2106.13314.pdf).
    """

    def __init__(self, train: bool = True):
        num_samples = 2000 if train else 1000
        points = 2 * torch.randn(num_samples, 3)
        self.data = torch.cat([
            torch.sin(points) + points,
            torch.cos(points) + points,
            (points ** 2).sum(dim=-1).unsqueeze(-1),
        ], dim=-1)
        self.concepts = (points > 0).float()
        self.targets = (self.concepts.sum(dim=-1) > 1).long()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return (self.data[idx], self.concepts[idx]), self.targets[idx]


class DatasetE(Dataset):
    """
    See "Promises and Pitfalls of Black-Box Concept Learning Models"
    Appendix E (https://arxiv.org/pdf/2106.13314.pdf).
    """

    def __init__(self, train: bool = True):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        mnist = datasets.MNIST(
            root='./data', train=train, transform=transform, download=True)

        # Filter digits 1-6
        idx = torch.cat([
            torch.argwhere(mnist.targets == digit)[:, 0]
            for digit in range(1, 7)
        ], dim=0)

        self.data, self.targets = mnist.data[idx].float(), mnist.targets[idx]
        self.concepts = torch.stack([
            self.targets == digit for digit in (1, 2, 3)], dim=-1).float()
        self.targets = (self.targets < 4).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return (self.data[idx], self.concepts[idx]), self.targets[idx]
