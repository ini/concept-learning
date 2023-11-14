from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import ray

from pathlib import Path
from ray import tune
from ray.tune import ResultGrid
from typing import Iterable, Union

from evaluate import evaluate
from lightning_ray import filter_results, group_results


### Typing

Results = Union[Iterable[ray.train.Result], dict[str, "Results"]]


### Helper Functions


def format_plot_title(plot_key: str | tuple[str] | tuple[str]) -> str:
    """
    Get a nicely-formatted title for the given dataset.

    Parameters
    ----------
    plot_key : str or tuple[str]
        Plot key to format
    """
    if isinstance(plot_key, tuple):
        if len(plot_key) > 1:
            return tuple(format_plot_title(key) for key in plot_key)
        else:
            plot_key = plot_key[0]

    plot_key = plot_key.replace("_", " ").title()
    plot_key = plot_key.replace("Mnist", "MNIST")
    plot_key = plot_key.replace("Cifar", "CIFAR")
    plot_key = plot_key.replace("Cub", "CUB")

    return plot_key


### Plotting


def plot_negative_interventions(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot negative intervention results.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    groupby = groupby[0] if len(groupby) == 1 else groupby
    for key, results in group_results(plot_results, groupby=groupby).items():
        results = group_results(results, groupby="eval_mode")
        if "neg_intervention" not in results:
            print("No negative intervention results found for:", key)
            continue

        accuracies = np.mean(
            np.stack(
                [
                    result.metrics["neg_intervention_accs"]["y"]
                    for result in results["neg_intervention"]
                ]
            ),
            axis=0,
        )
        plt.plot(np.linspace(0, 1, len(accuracies)), 1 - accuracies, label=key)

    plt.xlabel("Fraction of Concepts Intervened")
    plt.ylabel("Classification Error")
    plt.title(f"Negative Interventions: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_dir / f"{name}_{plot_key}_neg_intervention.png")
    if show:
        plt.show()
    plt.clf()


def plot_positive_interventions(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot negative intervention results.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    groupby = groupby[0] if len(groupby) == 1 else groupby
    for key, results in group_results(plot_results, groupby=groupby).items():
        results = group_results(results, groupby="eval_mode")
        if "pos_intervention" not in results:
            print("No positive intervention results found for:", key)
            continue

        accuracies = np.mean(
            np.stack(
                [
                    result.metrics["pos_intervention_accs"]["y"]
                    for result in results["pos_intervention"]
                ]
            ),
            axis=0,
        )
        plt.plot(np.linspace(0, 1, len(accuracies)), accuracies, label=key)

    plt.xlabel("Fraction of Concepts Intervened")
    plt.ylabel("Classification Accuracy")
    plt.title(f"Positive Interventions: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_dir / f"{name}_{plot_key}_pos_intervention.png")
    if show:
        plt.show()
    plt.clf()


def plot_random_concepts_residual(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot results with randomized concepts and residuals.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    baseline_accuracies = []
    random_concept_accuracies = []
    random_residual_accuracies = []

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    keys = sorted(plot_results.keys())
    info = (
        (baseline_accuracies, "accuracy", "test_acc"),
        (random_concept_accuracies, "random_concepts", "random_concept_acc"),
        (random_residual_accuracies, "random_residual", "random_residual_acc"),
    )
    for key in keys:
        results = group_results(plot_results[key], groupby="eval_mode")
        for collection, eval_mode, metric in info:
            collection.append(
                np.mean([result.metrics[metric] for result in results[eval_mode]])
            )

    # Plot
    x = np.arange(len(keys))
    plt.bar(x - 0.25, baseline_accuracies, label="Baseline", width=0.25)
    plt.bar(x, random_concept_accuracies, label="Random Concepts", width=0.25)
    plt.bar(x + 0.25, random_residual_accuracies, label="Random Residual", width=0.25)

    y_min = min(
        [
            *baseline_accuracies,
            *random_concept_accuracies,
            *random_residual_accuracies,
        ]
    )
    plt.xticks(np.arange(len(keys)), keys)
    plt.ylim(max(0, y_min - 0.1), 1)
    plt.ylabel("Classification Accuracy")
    plt.title(f"Random Concepts & Residual: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_dir / f"{name}_{plot_key}_random.png")
    if show:
        plt.show()
    plt.clf()


def plot_disentanglement(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot disentanglement metrics.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    keys = sorted(plot_results.keys())
    for key in keys:
        results = group_results(plot_results[key], groupby="eval_mode")
        if "correlation" not in results:
            print("No correlation results found for:", key)
            continue
        if "mutual_info" not in results:
            print("No mutual information results found for:", key)
            continue

        correlation = np.mean(
            [
                result.metrics["mean_abs_cross_correlation"]
                for result in results["correlation"]
            ]
        )
        mutual_info = np.mean(
            [result.metrics["mutual_info"] for result in results["mutual_info"]]
        )
        plt.scatter(correlation, mutual_info, label=key)

    plt.xlabel("Mean Absolute Cross-Correlation")
    plt.ylabel("Mutual Information")
    plt.title(f"Disentanglement Metrics: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_dir / f"{name}_{plot_key}_disentanglement.png")
    if show:
        plt.show()
    plt.clf()


def plot_mi_vs_intervention(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot mutual information vs positive intervention accuracy.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    keys = sorted(plot_results.keys())

    baseline_accuracies = []
    random_concept_accuracies = []
    random_residual_accuracies = []
    info = (
        (baseline_accuracies, "accuracy", "test_acc"),
        (random_concept_accuracies, "random_concepts", "random_concept_acc"),
        (random_residual_accuracies, "random_residual", "random_residual_acc"),
    )
    cross_corr = []
    mi_list = []
    pos_intervention = []

    for key in keys:
        results = group_results(plot_results[key], groupby="eval_mode")
        if "correlation" not in results:
            print("No correlation results found for:", key)
            continue
        if "mutual_info" not in results:
            print("No mutual information results found for:", key)
            continue

        for collection, eval_mode, metric in info:
            collection.append(
                np.mean([result.metrics[metric] for result in results[eval_mode]])
            )

        correlation = np.mean(
            [
                result.metrics["mean_abs_cross_correlation"]
                for result in results["correlation"]
            ]
        )

        accuracies = np.mean(
            np.stack(
                [
                    result.metrics["pos_intervention_accs"]["y"]
                    for result in results["pos_intervention"]
                ]
            ),
            axis=0,
        )[-1]
        pos_intervention.append(accuracies)

        cross_corr.append(correlation)
        mutual_info = np.mean(
            [result.metrics["mutual_info"] for result in results["mutual_info"]]
        )
        mi_list.append(mutual_info)

    for x_axis_type in ["cross_corr", "mi_list"]:
        for y_axis_type in [
            "baseline_accuracies",
            "random_concept_accuracies",
            "random_residual_accuracies",
            "pos_intervention",
        ]:
            x_axis_list = eval(x_axis_type)
            y_axis_list = eval(y_axis_type)
            plt.scatter(x_axis_list, y_axis_list)
            plt.xlabel(f"{x_axis_type}")
            plt.ylabel(f"{y_axis_type}")
            plt.title(f": {format_plot_title((x_axis_type, y_axis_type))} {name}")
            plt.legend()
            plt.savefig(save_dir / f"{name}_{x_axis_type}_{y_axis_type}_vs.png")
            if show:
                plt.show()
            plt.clf()


if __name__ == "__main__":
    PLOT_FUNCTIONS = {
        "neg_intervention": plot_negative_interventions,
        "pos_intervention": plot_positive_interventions,
        "random": plot_random_concepts_residual,
        "disentanglement": plot_disentanglement,
        "plot_mi_vs_intervention": plot_mi_vs_intervention,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        help="Experiment directory",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=PLOT_FUNCTIONS.keys(),
        help="Plot modes",
    )
    parser.add_argument(
        "--plotby",
        nargs="+",
        default=["dataset"],
        help=(
            "Config keys to group plots by "
            "(e.g. `--plotby dataset model_type` creates separate plots "
            "for each (dataset, model_type) combination)"
        ),
    )
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["model_type"],
        help="Config keys to group results on each plot by",
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob("**/eval/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load evaluation results
    tuner = tune.Tuner.restore(str(experiment_path / "eval"), trainable=evaluate)
    results = group_results(tuner.get_results(), groupby=args.plotby)

    # Plot results
    for plot_key, plot_results in results.items():
        for mode in args.mode:
            PLOT_FUNCTIONS[mode](
                plot_results,
                plot_key,
                groupby=args.groupby,
                save_dir=experiment_path / "plots",
            )
