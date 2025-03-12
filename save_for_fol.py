from __future__ import annotations
import os
import copy
import json

# make sure slurm isn't exposed
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]

if "SLURM_JOB_NAME" in os.environ:
    del os.environ["SLURM_JOB_NAME"]
from datasets.cxr import MIMIC_CXR
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
import argparse
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import ray.train

from copy import deepcopy
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray import air, tune
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import DATASET_INFO
from lightning_ray import LightningTuner
from nn_extensions import Chain
from models import ConceptLightningModel
from models.mutual_info import MutualInformationLoss
from models.posthoc_concept_pred import (
    ConceptResidualConceptPred,
    ConceptEmbeddingConceptPred,
)
from train import make_concept_model, make_datamodule
from utils import cross_correlation, set_cuda_visible_devices

from torchvision import transforms

### Interventions
from tqdm import tqdm


# def save_for_fol(model: ConceptLightningModel, train_loader: DataLoader, save_path: str) -> float:
#     """
#     Save concepts, residuals and labels to single numpy files.
    
#     Parameters
#     ----------
#     model : ConceptLightningModel
#         Model to evaluate
#     train_loader : DataLoader
#         Train data loader
#     save_path : str
#         Path to save the data
#     """
#     # Pre-allocate lists to collect batches
#     all_concept_residuals = []
#     all_targets = []
#     all_idxs = []
    
#     # Collect all data in memory
#     for (data, concepts, idxs), target in tqdm(train_loader, desc="Processing batches"):
#         _, residual, _ = model(data, concepts=concepts)
#         print(residual.device)
#         concept_residual = torch.cat([concepts, residual], dim=1)
        
#         # Move to CPU and convert to numpy
#         all_concept_residuals.append(concept_residual.cpu().numpy())
#         all_targets.append(target.cpu().numpy())
#         all_idxs.extend(idxs.cpu().numpy())
    
#     # Convert lists to arrays
#     all_concept_residuals = np.concatenate(all_concept_residuals, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)
#     all_idxs = np.array(all_idxs)
    
#     # Save all data in single files
#     np.save(os.path.join(save_path, 'concept_residuals.npy'), all_concept_residuals)
#     np.save(os.path.join(save_path, 'labels.npy'), all_targets)
#     np.save(os.path.join(save_path, 'indices.npy'), all_idxs)
def save_for_fol(model: ConceptLightningModel, train_loader: DataLoader, save_path: str) -> float:
    """
    Save concepts, residuals and labels to single numpy files.
    Ensures proper GPU usage for model and data processing.
    
    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    train_loader : DataLoader
        Train data loader
    save_path : str
        Path to save the data
    """
    # Ensure model is on GPU and in eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Pre-allocate lists to collect batches
    all_concept_residuals = []
    all_targets = []
    all_idxs = []
    
    # Collect all data in memory
    with torch.no_grad():  # Add no_grad for inference
        for (data, concepts, idxs), target in tqdm(train_loader, desc="Processing batches"):
            # Move input data to GPU
            data = data.to(device)
            concepts = concepts.to(device)
            target = target.to(device)
            
            # Forward pass
            _, residual, _ = model(data, concepts=concepts)
            concept_residual = torch.cat([concepts, residual], dim=1)
            
            # Move to CPU and convert to numpy
            all_concept_residuals.append(concept_residual.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            all_idxs.extend(idxs.detach().cpu().numpy())
    
    # Convert lists to arrays
    all_concept_residuals = np.concatenate(all_concept_residuals, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_idxs = np.array(all_idxs)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save all data in single files
    np.save(os.path.join(save_path, 'concept_residuals.npy'), all_concept_residuals)
    np.save(os.path.join(save_path, 'labels.npy'), all_targets)
    np.save(os.path.join(save_path, 'indices.npy'), all_idxs)
### Loading & Execution


def filter_eval_configs(configs: list[dict]) -> list[dict]:
    """
    Filter evaluation configs.

    Parameters
    ----------
    configs : list[dict]
        List of evaluation configs
    """
    configs_to_keep = []
    for config in configs:
        if config["model_type"] == "concept_whitening":
            if config["eval_mode"].endswith("intervention"):
                print("Interventions not supported for concept whitening models")
                continue

        if config["model_type"] == "no_residual" or config["residual_dim"] == 0:
            if config["eval_mode"] in ("correlation", "mutual_info"):
                print("Correlation / MI metrics not available for no-residual models")
                continue

        configs_to_keep.append(config)

    return configs_to_keep


def evaluate(config: dict):
    """
    Evaluate a trained model.

    Parameters
    ----------
    config : dict
        Evaluation configuration dictionary
    """
    metrics = {}

    # # Get data loader
    # if config["eval_mode"] == "concept_change_probe" and (config["dataset"] == "celeba" or config["dataset"] == "pitfalls_synthetic"):
    #     new_config = copy.deepcopy(config)
    #     new_config["num_concepts"] = 8
    #     train_loader = make_datamodule(**new_config).train_dataloader()
    #     val_loader = make_datamodule(**new_config).val_dataloader()
    #     test_loader = make_datamodule(**new_config).test_dataloader()
    # else:
    #     train_loader = make_datamodule(**config).train_dataloader()
    #     val_loader = make_datamodule(**config).val_dataloader()
    #     test_loader = make_datamodule(**config).test_dataloader()
    class IndexedMIMIC_CXR(MIMIC_CXR):
        def __getitem__(self, idx):
            (img, concepts), label = super().__getitem__(idx)
            return (img, concepts, idx), label
    transform_test = transforms.Compose(
                [
                    transforms.Resize(256),  # Resize shorter side to 256 pixels
                    transforms.CenterCrop(224),  # Center crop to 224x224
                    transforms.ToTensor(),  # Converts image to tensor
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # Normalization for ImageNet
                ]
            )

    # Create datasets
    train_dataset = IndexedMIMIC_CXR(
        root=config['data_dir'],
        split='train',
        subset=config['subset'],
        transform=transform_test
    )

    val_dataset = IndexedMIMIC_CXR(
        root=config['data_dir'],
        split='val',
        subset=config['subset'],
        transform=transform_test
    )

    test_dataset = IndexedMIMIC_CXR(
        root=config['data_dir'],
        split='test',
        subset=config['subset'],
        transform=transform_test
    )

    # Create Lightning DataModule
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_cpus']
    )

    train_loader = datamodule.train_dataloader()

    # Load model
    tuner = LightningTuner("val_acc", "max")
    model = tuner.load_model(make_concept_model, config["train_result"])
    if config["dataset"] == "mimic_cxr":
        dataset_info = DATASET_INFO[config["dataset"]][config["subset"]]
    else:
        dataset_info = DATASET_INFO[config["dataset"]]
    # Construct save path similar to how plot functions do it
    experiment_path = Path(config["train_result"].path).parent.parent
    fol_folder = "fol_processing"
    
    # Use the same pattern as in get_save_path from the first file
    plot_key = (config['dataset'], config['subset'], config['model_type'], str(config['residual_dim']))
    items = [str(key).replace(".", "_") for key in plot_key]
    
    save_dir = experiment_path / fol_folder
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "_".join(items)
    save_path.mkdir(exist_ok=True, parents=True)

    if config["eval_mode"] == "save_for_fol":
        save_for_fol(model, train_loader, save_path)

    # Report evaluation metrics
    ray.train.report(metrics)


if __name__ == "__main__":
    MODES = [
        "save_for_fol",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        help="Experiment directory",
    )
    parser.add_argument("--mode", nargs="+", default=MODES, help="Evaluation modes")
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["dataset", "model_type"],
        help="Config keys to group by when selecting best trial results",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all trained models (instead of best trial per group)",
    )
    parser.add_argument(
        "--num-cpus", type=float, default=1, help="Number of CPUs to use (per model)"
    )
    parser.add_argument(
        "--num-gpus", type=float, default=1, help="Number of GPUs to use (per model)"
    )
    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob("**/train/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load train results
    train_folder = "train"
    print("Loading training results from", experiment_path / train_folder)
    tuner = LightningTuner.restore(experiment_path / train_folder)
    print(tuner)
    print(experiment_path / train_folder)
    if args.all:
        results = tuner.get_results()
    else:
        results = [
            group.get_best_result()
            for group in tuner.get_results(groupby=args.groupby).values()
        ]

    # Create evaluation configs
    results = [result for result in results if result.config is not None]
    eval_configs = filter_eval_configs(
        [
            {
                **result.config["train_loop_config"],
                "train_result": result,
                "eval_mode": mode,
            }
            for result in results
            for mode in args.mode
        ]
    )

    # Get available resources
    if args.num_gpus < 1:
        set_cuda_visible_devices(available_memory_threshold=args.num_gpus)

    # Run evaluations
    eval_folder = "fol_processing"
    tuner = tune.Tuner(
        tune.with_resources(
            evaluate,
            resources={
                "cpu": args.num_cpus,
                "gpu": args.num_gpus if torch.cuda.is_available() else 0,
            },
        ),
        param_space=tune.grid_search(eval_configs),
        run_config=air.RunConfig(name=eval_folder, storage_path=experiment_path),
    )
    eval_results = tuner.fit()
