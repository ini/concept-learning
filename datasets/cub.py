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



class CUB(Dataset):
    """
    Caltech-UCSD Birds-200-2011 dataset (CUB-200-2011).
    See https://www.vision.caltech.edu/datasets/cub_200_2011/ for more info.

    The dataset has 200 classes and 11,788 images.
    Each sample includes an image and a one-hot vector of 312 binary attributes.
    """

    def __init__(
        self,
        root: str,
        split: Literal['train', 'val', 'test'] = 'train',
        transform: Callable | None = None,
        download: bool = False,
        use_attribute_uncertainty: bool = False):
        """
        Parameters
        ----------
        root : str
            Root directory of dataset
        split : one of {'train', 'val', 'test'}
            The dataset split to use
        transform : Callable, optional
            A function / transform that takes in an PIL image and returns a
            transformed version (e.g. `torchvision.transforms.RandomCrop`)
        download : bool, default=False
            Whether to download the dataset if it is not found in root
        """
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.use_attribute_uncertainty = use_attribute_uncertainty
        self.dataset_dir = (Path(root) / self.__class__.__name__).resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Check if data already exists
        resource_paths = (
            self.dataset_dir / 'CUB_200_2011',
            self.dataset_dir / 'CUB_processed/class_attr_data_10/train.pkl',
            self.dataset_dir / 'CUB_processed/class_attr_data_10/train.pkl',
            self.dataset_dir / 'CUB_processed/class_attr_data_10/train.pkl',
        )
        if not all(path.exists() for path in resource_paths):
            # Download data
            if download:
                self.download()
            else:
                raise RuntimeError(
                    'Dataset not found. You can use download=True to download it.')

            # Process data
            processed_data_paths = [
                self.dataset_dir / f'CUB_processed/class_attr_data_10/{split}.pkl'
                for split in ('train', 'val', 'test')
            ]
            if any(not path.exists() for path in processed_data_paths):
                self.process_data()

        # Load data
        split_path = self.dataset_dir / f'CUB_processed/class_attr_data_10/{split}.pkl'
        with open(split_path, 'rb') as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        attributes = torch.as_tensor(img_data['attribute_label']).float()
        target = img_data['class_label']

        return (img, attributes), target

    def download(self):
        """
        Download original CUB-200-2011 dataset.
        """
        URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
        download_and_extract_archive(URL, self.dataset_dir, filename='CUB_200_2011.tgz')

    def process_data(self):
        """
        Process data and generate train / val / test splits
        (see https://github.com/yewsiang/ConceptBottleneck/).
        """
        data_dir = self.dataset_dir / 'CUB_200_2011'
        print(f'Processing data from {data_dir}')

        # Map from full image path to image id
        img_path_to_id = {}
        with open(data_dir / 'images.txt', 'r') as file:
            for line in file:
                img_id, img_path = line.strip().split()
                img_path_to_id[data_dir / 'images' / img_path] = int(img_id)

        # Map from image id to a list of attribute labels
        attribute_labels_all = defaultdict(list)

        # Map from image id to a list of attribute certainties
        attribute_certainties_all = defaultdict(list)

        # Map from image id to a list of attribute labels calibrated for uncertainty
        attribute_uncertain_labels_all = defaultdict(list)

        # Calibrate main label based on uncertainty label
        # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
        uncertainty_map = {
            1: {1: 0, 2: 0.5, 3: 0.75, 4: 1}, 
            0: {1: 0, 2: 0.5, 3: 0.25, 4: 0},
        }

        # Process image attribute labels
        with open(data_dir / 'attributes/image_attribute_labels.txt', 'r') as file:
            for line in file:
                items = [int(item) for item in line.strip().split()[:4]]
                img_id, attribute_idx, attribute_label, attribute_certainty = items
                uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
                attribute_labels_all[img_id].append(attribute_label)
                attribute_uncertain_labels_all[img_id].append(uncertain_label)
                attribute_certainties_all[img_id].append(attribute_certainty)

        # Get the official train / test split
        train_img_ids, test_img_ids = set(), set()
        with open(data_dir / 'train_test_split.txt', 'r') as file:
            for line in file:
                img_id, is_train = line.strip().split()
                if is_train == '1':
                    train_img_ids.add(int(img_id))
                else:
                    test_img_ids.add(int(img_id))

        # Hold out a validation set from the training set
        random.seed(42)
        val_ratio = 0.2
        val_img_ids = set(random.sample(
            list(train_img_ids), int(val_ratio * len(train_img_ids))))
        train_img_ids -= val_img_ids

        # Get image metadata
        train_data, val_data, test_data = [], [], []
        img_dirs = [path for path in (data_dir / 'images').iterdir() if path.is_dir()]
        img_dirs.sort() # sort by class index
        for i, img_dir in enumerate(img_dirs):
            for img_path in img_dir.iterdir():
                img_id = img_path_to_id[img_path]
                metadata = {
                    'id': img_id,
                    'img_path': img_path,
                    'class_label': i,
                    'attribute_label': torch.tensor(
                        attribute_labels_all[img_id]),
                    'attribute_certainty': torch.tensor(
                        attribute_certainties_all[img_id]),
                    'uncertain_attribute_label': torch.tensor(
                        attribute_uncertain_labels_all[img_id]),
                }
                if img_id in train_img_ids:
                    train_data.append(metadata)
                elif img_id in val_img_ids:
                    val_data.append(metadata)
                else:
                    test_data.append(metadata)

        # Save processed dataset
        save_dir = self.dataset_dir / 'CUB_processed/class_attr_data_10'
        save_dir.mkdir(parents=True, exist_ok=True)
        for dataset in ('train','val','test'):
            with open(save_dir / f'{dataset}.pkl', 'wb') as file:
                if dataset == 'train':
                    pickle.dump(train_data, file)
                elif dataset == 'val':
                    pickle.dump(val_data, file)
                else:
                    pickle.dump(test_data, file)
