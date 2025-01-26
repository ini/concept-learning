from __future__ import annotations

import pickle
import random
import torch

from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from typing import Callable, Literal
from torchvision import datasets
import numpy as np

import h5py


class AA2(Dataset):
    """
    AA2 Dataset
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
        download: bool = False,
        *args,
        **kwargs,
    ):
        self.root = root

        self.dataset_dir = (Path(root) / self.__class__.__name__).resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.split = split
        self.data = h5py.File(self.image_data_path, "r")[split]

        all_data = self.data[0]
        img = all_data[:512*512*3].reshape(3, 512, 512)
        label = all_data[512*512*3]
        concepts = all_data[512*512*3+1:]

        self.num_concepts = concepts.shape[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_data = self.data[idx]
        img = all_data[:512*512*3].reshape(3, 512, 512)
        label = all_data[512*512*3]
        concepts = all_data[512*512*3+1:]
        img = torch.from_numpy(img).float()
        if self.transform is not None:
            img = self.transform(img)
        
        return (img, concepts), label
