from __future__ import annotations

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
from train import make_concept_model, make_datamodule
from utils import cross_correlation, set_cuda_visible_devices


### Interventions


def test_mutual_info(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    num_mi_epochs: int = 5,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_mi_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    (data, concepts), targets = next(iter(test_loader))
    _, residual, _ = model(data, concepts=concepts)
    concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
    mutual_info_estimator = MutualInformationLoss(residual_dim, concept_dim)

    # Learn mutual information estimator
    for epoch in range(num_mi_epochs):
        for (data, concepts), targets in test_loader:
            with torch.no_grad():
                _, residual, _ = model(data, concepts=concepts)
            mutual_info_estimator.step(residual, concepts)

    # Calculate mutual information
    mutual_infos = []
    for (data, concepts), target in test_loader:
        with torch.no_grad():
            _, residual, _ = model(data, concepts=concepts)
        mutual_infos.append(mutual_info_estimator(residual, concepts).item())

    return np.mean(mutual_infos)


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

    # Load model
    tuner = LightningTuner("val_acc", "max")
    model_path = tuner.load_model(
        make_concept_model, config["train_result"], return_path=True
    )

    config["base_model_ckpt"] = model_path
    del config["train_result"]
    run_config = tune.get_trial_context().run_config
    assert 0, run_config

    tuner.fit(
        make_concept_model,
        make_datamodule,
        param_space=config,
        save_dir=config.get("save_dir"),
        experiment_name=experiment_name,
        num_workers_per_trial=config.get("num_workers", 1),
        num_cpus_per_worker=config.get("num_cpus", 1),
        num_gpus_per_worker=config.get("num_gpus", 1),
        gpu_memory_per_worker=config.get("gpu_memory_per_worker", None),
        groupby=config.get("groupby", []),
    )


if __name__ == "__main__":
    MODES = [
        "accuracy",
        "neg_intervention",
        "pos_intervention",
        "random_concepts",
        "random_residual",
        "correlation",
        # "mutual_info",
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
    parser.add_argument(
        "--new-lr", type=float, default=4e-3, help="New learning rate for fine-tuning"
    )
    parser.add_argument(
        "--new-epochs", type=int, default=50, help="Number of epochs for fine-tuning"
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob("**/train/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load train results
    print("Loading training results from", experiment_path / "train")
    tuner = LightningTuner.restore(experiment_path / "train")
    results = tuner.get_results()

    # Create evaluation configs
    results = [result for result in results if result.config is not None]

    finetune_configs = filter_eval_configs(
        [
            {
                **result.config["train_loop_config"],
                "train_result": result,
            }
            for result in results
        ]
    )

    # Get available resources
    if args.num_gpus < 1:
        set_cuda_visible_devices(available_memory_threshold=args.num_gpus)

    # Run evaluations
    # tuner = tune.Tuner(
    #     tune.with_resources(
    #         evaluate,
    #         resources={
    #             "cpu": args.num_cpus,
    #             "gpu": args.num_gpus if torch.cuda.is_available() else 0,
    #         },
    #     ),
    #     param_space=tune.grid_search(finetune_configs),
    #     run_config=air.RunConfig(name="mixture", storage_path=experiment_path),
    # )
    # results = tuner.fit()
    config = finetune_configs[0]
    experiment_name = config.get("experiment_module_name").split(".")[-1]
    old_experiment_name = config["train_result"].path
    parts = old_experiment_name.split("/")
    index = parts.index(experiment_name)
    data_index = index + 1
    date = parts[data_index]
    experiment_name = f"{experiment_name}/{date}/mixture"

    # filter out the train_resul model path, we want to load that
    tuner = LightningTuner("val_acc", "max")
    for config in finetune_configs:
        train_result = config["train_result"]
        model_path = tuner.load_model(
            make_concept_model, config["train_result"], return_path=True
        )
        config["base_model_ckpt"] = model_path
        del config["train_result"]
        # update training mode
        config["training_mode"] = "semi_independent"
        config["lr"] = args.new_lr
        config["max_epochs"] = args.new_epochs

    tuner = LightningTuner(
        metric="val_acc",
        mode="max",
        scheduler=None,
        num_samples=1,
    )
    tuner.fit(
        make_concept_model,
        make_datamodule,
        param_space=tune.grid_search(finetune_configs),
        save_dir=config.get("save_dir"),
        experiment_name=experiment_name,
        num_workers_per_trial=config.get("num_workers", 1),
        num_cpus_per_worker=config.get("num_cpus", 1),
        num_gpus_per_worker=config.get("num_gpus", 1),
        gpu_memory_per_worker=config.get("gpu_memory_per_worker", None),
        groupby=config.get("groupby", []),
    )
