from __future__ import annotations

import argparse
import importlib
import os
import pytorch_lightning as pl
import torch

from datetime import datetime
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray import tune
from ray.air import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer
from ray.tune import TuneConfig, Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.utils.data import DataLoader
from typing import Any

from loader import get_concept_loss_fn, get_data_loaders, DATASET_INFO
from models import *
from ray_utils import config_get, config_set, GroupScheduler, RayCallback
from utils import cross_correlation, set_cuda_visible_devices


def get_train_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Get Ray experiment configuration dictionary from command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Load provided experiment config
    experiment_module = importlib.import_module(args.config)
    config = experiment_module.get_config()
    config_set(config, "experiment_module_name", args.config)

    # Override config with command line arguments
    for key, value in vars(args).items():
        if isinstance(value, list):
            config_set(config, key, tune.grid_search(value))
        elif value is not None:
            config_set(config, key, value)

    # Use absolute paths
    config_set(
        config, "data_dir", Path(config_get(config, "data_dir")).expanduser().resolve()
    )
    config_set(
        config, "save_dir", Path(config_get(config, "save_dir")).expanduser().resolve()
    )

    return config


def make_concept_model(
    loader: DataLoader | None = None, **config
) -> ConceptLightningModel:
    """
    Create a concept model.

    Parameters
    ----------
    loader : DataLoader or None
        Data loader to use for dummy pass
    experiment_module_name : str
        Name of the experiment module (e.g. 'experiments.cifar')
    model_type : str
        Model type
    training_mode : one of {'independent', 'sequential', 'joint'}
        Training mode (see https://arxiv.org/abs/2007.04612)
    device : str
        Device to load model on
    concept_dim : int
        Size of concept vector
    residual_dim : int
        Size of residual vector
    mi_estimator_hidden_dim : int
        Hidden dimension of mutual information estimator
    mi_optimizer_lr : float
        Learning rate of mutual information estimator optimizer
    whitening_alignment_frequency : int
        Frequency of concept alignment for whitening (in epochs)
    """
    experiment_module = importlib.import_module(config["experiment_module_name"])
    model_type = config["model_type"]

    # Update config with any missing dataset information (e.g. concept_dim, num_classes)
    dataset_info = DATASET_INFO[config["dataset"]]
    config = {**dataset_info, **config}

    # Get concept loss function
    config["concept_loss_fn"] = get_concept_loss_fn(config["dataset"])

    # No residual
    if model_type == "no_residual":
        config = {**config, "residual_dim": 0}
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    # With latent residual
    elif model_type in "latent_residual":
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    # With decorrelated residual
    elif model_type == "decorrelated_residual":
        residual_loss_fn = lambda r, c: cross_correlation(r, c).square().mean()
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(
            model, residual_loss_fn=residual_loss_fn, **config
        )

    # With MI-minimized residual
    elif model_type == "mi_residual":
        model = experiment_module.make_concept_model(config)
        model = MutualInfoConceptLightningModel(model, **config)

    # With iterative normalization
    elif model_type == "iter_norm":
        config = {**config, "norm_type": "iter_norm"}
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    # With concept whitening
    elif model_type == "concept_whitening":
        config = {
            **config,
            "concept_type": "continuous",
            "norm_type": "concept_whitening",
            "training_mode": "joint",
        }
        model = experiment_module.make_concept_model(config)
        model = ConceptWhiteningLightningModel(model, **config)

    else:
        raise ValueError("Unknown model type:", model_type)
    if loader is not None:
        model.dummy_pass(loader)

    return model


def train_concept_model(config: dict[str, Any]):
    """
    Train a concept model.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    """
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(
        config["dataset"], data_dir=config["data_dir"], batch_size=config["batch_size"]
    )

    # Create model
    model = make_concept_model(loader=train_loader, **config)

    # Train model
    if MPSAccelerator.is_available():
        accelerator = "cpu"
    elif config["num_gpus"] > 0:
        accelerator = "gpu"
    else:
        accelerator = "auto"
    trainer = pl.Trainer(
        accelerator=accelerator,
        strategy=RayDDPStrategy(find_unused_parameters=True),
        devices="auto",
        logger=False,  # logging metrics is handled by Ray
        callbacks=[model.callback(), RayCallback(**config)],
        max_epochs=config["num_epochs"],
        enable_checkpointing=False,  # checkpointing is handled by Ray
        enable_progress_bar=False,
        plugins=[RayLightningEnvironment()],
        profiler="simple",
    )
    trainer.fit(model, train_loader, val_loader)


def get_ray_trainer(config: dict[str, Any] = {}) -> TorchTrainer:
    """
    Ray trainer for training concept models.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    """
    num_gpus = config_get(config, "num_gpus", 1) if torch.cuda.is_available() else 0
    return TorchTrainer(
        train_concept_model,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=(num_gpus > 0),
            resources_per_worker={"GPU": num_gpus} if num_gpus > 0 else None,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments.pitfalls",
        help="Experiment configuration module",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--num-gpus", type=float, help="Number of GPUs to use (per model)"
    )
    parser.add_argument("--data-dir", type=str, help="Directory where data is stored")
    parser.add_argument("--save-dir", type=str, help="Directory to save models to")
    parser.add_argument(
        "--dataset", type=str, choices=DATASET_INFO.keys(), help="Dataset to train on"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        nargs="+",
        choices=[
            "no_residual",
            "latent_residual",
            "decorrelated_residual",
            "mi_residual",
            "iter_norm",
            "concept_whitening",
        ],
        help="Model type",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        nargs="+",
        choices=["independent", "sequential", "joint"],
        help="Training mode",
    )
    parser.add_argument(
        "--residual-dim", type=int, nargs="+", help="Dimensionality of the residual"
    )
    parser.add_argument(
        "--num-epochs", type=int, nargs="+", help="Number of epochs to train for"
    )
    parser.add_argument("--lr", type=float, nargs="+", help="Learning rate")
    parser.add_argument("--batch-size", type=int, nargs="+", help="Batch size")
    parser.add_argument("--alpha", type=float, nargs="+", help="Weight of concept loss")
    parser.add_argument("--beta", type=float, nargs="+", help="Weight of residual loss")
    parser.add_argument(
        "--mi-estimator-hidden-dim",
        type=int,
        nargs="+",
        help="Hidden dimension of the MI estimator",
    )
    parser.add_argument(
        "--mi-optimizer-lr",
        type=float,
        nargs="+",
        help="Learning rate of the MI estimator optimizer",
    )
    parser.add_argument(
        "--cw-alignment-frequency",
        type=int,
        nargs="+",
        help="Frequency of whitening alignment",
    )
    parser.add_argument(
        "--checkpoint-frequency", type=int, nargs="+", help="Frequency of checkpointing"
    )
    parser.add_argument(
        "--tensorboard-port", type=int, default=0, help="Port to launch TensorBoard"
    )
    parser.add_argument(
        "--groupby", type=str, nargs="+", help="Config keys to group by"
    )

    args = parser.parse_args()
    config = get_train_config(args)

    # Download datasets (if necessary) before launching Ray Tune
    # Avoids each initial worker trying to downloading the dataset simultaneously
    dataset_names = config_get(config, "dataset")
    if isinstance(dataset_names, dict) and "grid_search" in dataset_names:
        dataset_names = list(dataset_names.values())
    dataset_names = [dataset_names] if isinstance(dataset_names, str) else dataset_names
    for dataset_name in dataset_names:
        get_data_loaders(dataset_name, data_dir=config_get(config, "data_dir"))

    # Get experiment name
    date = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_name = config_get(config, "experiment_module_name").split(".")[-1]
    experiment_name = f"{experiment_name}/{date}/train"

    # Set Ray storage directory
    os.environ.setdefault(
        "RAY_AIR_LOCAL_CACHE_DIR", str(config_get(config, "save_dir"))
    )

    # Create hyperparameter scheduler
    scheduler = None
    groupby = config_get(config, "groupby", None)
    if groupby is not None:
        scheduler = AsyncHyperBandScheduler(
            metric="val_acc", mode="max", max_t=config_get(config, "num_epochs")
        )
        scheduler = GroupScheduler(scheduler, groupby=config_get(config, "groupby"))

    # Launch TensorBoard
    port = config_get(config, "tensorboard_port")
    if port is not None:
        from tensorboard import program

        experiment_dir = Path(config_get(config, "save_dir")) / experiment_name
        experiment_dir = experiment_dir.parent
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", str(experiment_dir), "--port", str(port)])
        url = tb.launch()
        print(f"TensorBoard started at {url}", "\n")

    # Get available resources
    num_gpus = config_get(config, "num_gpus", 1) if torch.cuda.is_available() else 0
    if num_gpus < 1:
        set_cuda_visible_devices(available_memory_threshold=num_gpus)

    # Get available resources
    num_gpus = config_get(config, "num_gpus", 1) if torch.cuda.is_available() else 0
    if num_gpus < 1:
        set_cuda_visible_devices(available_memory_threshold=num_gpus)

    # Train the model(s)
    tuner = Tuner(
        get_ray_trainer(config),
        param_space={"train_loop_config": config},
        tune_config=TuneConfig(
            metric="val_acc", mode="max", scheduler=scheduler, num_samples=1
        ),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=config_get(config, "save_dir"),
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute="val_acc",
                checkpoint_score_order="max",
            ),
        ),
    )
    results = tuner.fit()
