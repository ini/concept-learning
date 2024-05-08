from __future__ import annotations
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
from train import make_concept_model, make_datamodule
from utils import cross_correlation, set_cuda_visible_devices


### Interventions


class Randomize(nn.Module):
    """
    Shuffle data along the batch dimension.
    """

    def forward(self, x: Tensor, *args, **kwargs):
        return x[torch.randperm(len(x))]


class Intervention(nn.Module):
    """
    Intervene on a random subset of concepts.
    """

    def __init__(self, num_interventions: int, negative: bool = False):
        """
        Parameters
        ----------
        num_interventions : int
            Number of concepts to intervene on
        negative : bool
            Whether to intervene with incorrect concept values
        """
        super().__init__()
        self.num_interventions = num_interventions
        self.negative = negative

    def forward(self, x: Tensor, concepts: Tensor, which=None):
        if self.negative:
            concepts = 1 - concepts  # flip binary concepts to opposite values

        concept_dim = concepts.shape[-1]
        idx = torch.randperm(concept_dim)[: self.num_interventions]
        x[..., idx] = concepts[..., idx]
        if which is None:
            return x
        else:
            new_which = torch.ones_like(which, device=x.device)
            new_which[..., idx] = 0

            # avg = 1 - new_which.float().mean(-1).mean()
            # assert 0, f"Average number of concepts intervened on: {avg}"
            return x, concepts, new_which


### Evaluations
from lightning.pytorch.plugins import LightningEnvironment


def test(model: pl.LightningModule, loader: DataLoader):
    """
    Test model.

    Parameters
    ----------
    model : pl.LightningModule
        Model to test
    loader : DataLoader
        Test data loader
    """
    trainer = pl.Trainer(
        accelerator="cpu" if MPSAccelerator.is_available() else "auto",
        enable_progress_bar=False,
        plugins=[LightningEnvironment()],
    )
    return trainer.test(model, loader)[0]


def test_interventions(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    concept_dim: int,
    negative: bool,
    max_samples: int = 10,
) -> float:
    """
    Test model accuracy with concept interventions.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Dimension of concept vector
    negative : bool
        Whether to intervene with incorrect concept values
    max_samples : int
        Maximum number of interventions to test (varying the # of concepts intervened on)
    """
    x = np.linspace(
        0, concept_dim + 1, num=min(concept_dim + 2, max_samples), dtype=int
    )
    # x = x[::-1]
    y = np.zeros(len(x))
    for i, num_interventions in enumerate(x):
        intervention = Intervention(num_interventions, negative=negative)
        new_model = deepcopy(model)

        if new_model.concept_model.mixer is not None:
            new_model.concept_model.mixer = Chain(
                intervention,
                new_model.concept_model.mixer,
            )
        else:
            new_model.concept_model.target_network = Chain(
                intervention,
                new_model.concept_model.target_network,
            )
        results = test(new_model, test_loader)
        y[i] = results["test_acc"]

    return {"x": x, "y": y}


def test_random_concepts(
    model: ConceptLightningModel, test_loader: DataLoader
) -> float:
    """
    Test model accuracy with randomized concept predictions.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    # Shuffle data
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=True,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory,
    )

    new_model = deepcopy(model)
    new_model.concept_model.concept_network = Chain(
        new_model.concept_model.concept_network, Randomize()
    )
    results = test(new_model, test_loader)
    return results["test_acc"]


def test_random_residual(
    model: ConceptLightningModel, test_loader: DataLoader
) -> float:
    """
    Test model accuracy with randomized residual values.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    # Shuffle data
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=True,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory,
    )

    new_model = deepcopy(model)
    new_model.concept_model.residual_network = Chain(
        new_model.concept_model.residual_network, Randomize()
    )
    results = test(new_model, test_loader)
    return results["test_acc"]


def test_correlation(model: ConceptLightningModel, test_loader: DataLoader) -> float:
    """
    Test mean absolute cross correlation between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    correlations = []
    for (data, concepts), target in test_loader:
        with torch.no_grad():
            _, residual, _ = model(data, concepts=concepts)
        correlations.append(cross_correlation(concepts, residual).abs().mean().item())

    return np.mean(correlations)


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

    # Get data loader
    test_loader = make_datamodule(**config).test_dataloader()

    # Load model
    tuner = LightningTuner("val_acc", "max")
    model = tuner.load_model(make_concept_model, config["train_result"])

    # Evaluate model
    if config["eval_mode"] == "accuracy":
        results = test(model, test_loader)
        for key in ("test_acc", "test_concept_acc"):
            if key in results:
                metrics[key] = results[key]

    if config["eval_mode"] == "neg_intervention":
        concept_dim = DATASET_INFO[config["dataset"]]["concept_dim"]
        metrics["neg_intervention_accs"] = test_interventions(
            model, test_loader, concept_dim, negative=True
        )

    elif config["eval_mode"] == "pos_intervention":
        concept_dim = DATASET_INFO[config["dataset"]]["concept_dim"]
        metrics["pos_intervention_accs"] = test_interventions(
            model, test_loader, concept_dim, negative=False
        )

    elif config["eval_mode"] == "random_concepts":
        metrics["random_concept_acc"] = test_random_concepts(model, test_loader)

    elif config["eval_mode"] == "random_residual":
        metrics["random_residual_acc"] = test_random_residual(model, test_loader)

    elif config["eval_mode"] == "correlation":
        metrics["mean_abs_cross_correlation"] = test_correlation(model, test_loader)

    elif config["eval_mode"] == "mutual_info":
        metrics["mutual_info"] = test_mutual_info(model, test_loader)

    # Report evaluation metrics
    ray.train.report(metrics)


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
        "--evaluate-mixer", action="store_true", help="Evaluate mixer models"
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob("**/train/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load train results
    train_folder = "train" if not args.evaluate_mixer else "train_mixer"
    print("Loading training results from", experiment_path / train_folder)
    tuner = LightningTuner.restore(experiment_path / train_folder)
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
    eval_folder = "eval" if not args.evaluate_mixer else "eval_mixer"
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
