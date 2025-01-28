import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Literal, Callable
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
import os
import multiprocessing as mp
import h5py

def process_sample(args) -> None:
    """
    Process a single sample from the dataset
    """
    h5_path, img_dir, metadata_dir, idx, split_name = args
    
    # Get first digit for directory organization
    first_digit = str(idx)[0]
    
    # Read data for this index only
    with h5py.File(h5_path, 'r') as f:
        data = f[split_name][idx]
    
    # Process image
    img = data[:512*512*3].reshape(3, 512, 512)
    img_uint8 = (img / 256).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8.transpose(1, 2, 0))
    resize_transform = transforms.Resize(256)
    img_resized = resize_transform(img_pil)
    img_final = np.array(img_resized, dtype=np.uint8).transpose(2, 0, 1)
    
    # Save image with original index in digit-based directory
    img_path = os.path.join(img_dir, first_digit, f"{idx:08d}.npy")
    if not os.path.exists(img_path):
        np.save(img_path, img_final)
    
    # Save metadata with original index in digit-based directory
    metadata = data[512*512*3:]
    metadata_path = os.path.join(metadata_dir, first_digit, f"{idx:08d}.npy")
    np.save(metadata_path, metadata)

def convert_h5_to_numpy(h5_path: str, output_root: str, subset: str, split: str):
    """
    Converts H5 dataset to numpy files organized by split first, then subset
    Files are organized in directories based on their first index digit
    """
    # Replace the data path prefix with home directory
    output_root = str(output_root).replace('/data/Datasets', '/home/renos')
    output_root = Path(output_root)
    split_dir = output_root / split
    
    # Create directory structure
    img_dir = str(split_dir / 'images')
    metadata_dir = str(split_dir / 'metadata' / subset)
    
    # Create digit-based directories (0-9)
    for digit in range(10):
        os.makedirs(os.path.join(img_dir, str(digit)), exist_ok=True)
        os.makedirs(os.path.join(metadata_dir, str(digit)), exist_ok=True)
    
    # Get dataset size and split name
    with h5py.File(h5_path, 'r') as f:
        split_name = list(f.keys())[0]
        total_samples = len(f[split_name])
    
    print(f"Converting {total_samples} samples for {split}/{subset}...")
    
    # Prepare argument tuples without loading the data
    process_args = [(h5_path, img_dir, metadata_dir, idx, split_name) 
                   for idx in range(total_samples)]
    
    # Use multiprocessing to process samples
    num_cpus = 256
    with mp.Pool(num_cpus) as pool:
        list(tqdm(
            pool.imap(process_sample, process_args, chunksize=50),
            total=total_samples,
            desc="Processing samples"
        ))

class MIMIC_CXR(Dataset):
    """
    MIMIC CXR Dataset organized by split first, then subset
    Files are organized in directories based on their first index digit
    """
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        subset: str = "cardiomegaly",
        transform: Callable | None = None,
        convert_if_needed: bool = True,
        *args,
        **kwargs,
    ):
        self.root = Path(str(root).replace('/data/Datasets', '/home/renos'))
        self.split = "valid" if split == "val" else split
        self.subset = subset
        self.transform = transform
        
        # Setup paths
        self.h5_path = Path(str(root)) / subset / "dataset_g" / "dataset_g" / f"{self.split}.h5"
        
        # Split is the top-level directory
        self.split_dir = self.root / self.split
        self.img_dir = self.split_dir / 'images'
        self.metadata_dir = self.split_dir / 'metadata' / subset
        
        # Convert if needed
        if convert_if_needed and not self.metadata_dir.exists():
            print(f"Converting {self.split} set for {subset}...")
            convert_h5_to_numpy(str(self.h5_path), str(self.root), subset, self.split)
        
        # Get total number of samples from h5 file
        with h5py.File(self.h5_path, 'r') as f:
            split_name = list(f.keys())[0]
            self.total_samples = len(f[split_name])
        
        # Get number of concepts from first metadata file
        first_metadata = np.load(self.metadata_dir / "0" / f"{0:08d}.npy")
        self.num_concepts = len(first_metadata) - 1
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Get first digit for directory lookup
        first_digit = str(idx)[0]
        
        # Load data using the index from the appropriate directory
        img = np.load(self.img_dir / first_digit / f"{idx:08d}.npy")
        metadata = np.load(self.metadata_dir / first_digit / f"{idx:08d}.npy")
        
        # Split metadata into label and concepts
        label = metadata[0]
        concepts = metadata[1:]
        
        # Convert image for processing
        img = Image.fromarray(img.transpose(1, 2, 0))
        
        if self.transform is not None:
            img = self.transform(img)
            
        return (img, torch.tensor(concepts, dtype=torch.float32)), int(label)