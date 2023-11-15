from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from pathlib import Path
from ray import tune
from ray.train import Result
from ray.tune import ResultGrid
from typing import Callable

from evaluate import evaluate
from lightning_ray import group_results



### Helper Functions

def format_plot_title(plot_key: str | tuple[str, ...] | list[str]) -> str:
    """
    Get a nicely-formatted title for the given dataset.

    Parameters
    ----------
    plot_key : str or tuple[str]
        Plot key to format
    """
    if isinstance(plot_key, (list, tuple)):
        if len(plot_key) > 1:
            return tuple(format_plot_title(key) for key in plot_key)
        else:
            plot_key = plot_key[0]

    if isinstance(plot_key, str):
        plot_key = plot_key.replace("_", " ").title()
        plot_key = plot_key.replace("Mnist", "MNIST")
        plot_key = plot_key.replace("Cifar", "CIFAR")
        plot_key = plot_key.replace("Cub", "CUB")

    return plot_key

def get_save_path(
    plot_key: tuple,
    prefix: str | None = None,
    suffix: str | None = None,
    save_dir: Path | str = './plots',
) -> Path:
    """
    Get the save path for the given plot.
    """
    items = [str(key) for key in plot_key]
    if prefix:
        items.insert(0, prefix)
    if suffix:
        items.append(suffix)

    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    return save_dir / '_'.join(items)

def plot_curves(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str],
    title: str,
    x_label: str,
    y_label: str,
    get_x: Callable[[ResultGrid], np.ndarray],
    get_y: Callable[[Result], np.ndarray],
    eval_mode: str,
    save_dir: Path | str,
    save_name: str,
    prefix: str | None = None,
    show: bool = True,
):
    plt.clf()
    save_path = get_save_path(
        plot_key, prefix=prefix, suffix=save_name, save_dir=save_dir)

    y_values, columns = [], []
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    for key, results in plot_results.items():
        results = group_results(results, groupby='eval_mode')
        if eval_mode not in results:
            print(f"No {eval_mode} results found for:", key)
            continue

        x = get_x(results[eval_mode])
        y = np.stack([get_y(result) for result in results[eval_mode]]).mean(axis=0)
        plt.plot(x, y, label=key)
        y_values.append(y)
        columns.append(f'{key} {y_label}')

    # Create CSV file
    x = np.linspace(0, 1, len(y_values[0]))
    data = np.stack([x, *y_values], axis=1)
    columns.insert(0, x_label)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path.with_suffix('.csv'), index=False)

    # Create figure
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path.with_suffix('.png'))
    if show:
        plt.show()



### Plotting

def plot_negative_interventions(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
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
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plot_curves(
        plot_results,
        plot_key,
        groupby=groupby,
        title=f"Negative Interventions: {format_plot_title(plot_key)} {name}",
        x_label="Fraction of Concepts Intervened",
        y_label="Classification Error",
        get_x=lambda results: np.linspace(
            0, 1, len(results[0].metrics['neg_intervention_accs']['y'])),
        get_y=lambda result: 1 - result.metrics['neg_intervention_accs']['y'],
        eval_mode='neg_intervention',
        save_dir=save_dir,
        save_name='neg_intervention',
        prefix=name,
        show=show,
    )

def plot_positive_interventions(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot positive intervention results.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plot_curves(
        plot_results,
        plot_key,
        groupby=groupby,
        title=f"Positive Interventions: {format_plot_title(plot_key)} {name}",
        x_label="Fraction of Concepts Intervened",
        y_label="Classification Accuracy",
        get_x=lambda results: np.linspace(
            0, 1, len(results[0].metrics['pos_intervention_accs']['y'])),
        get_y=lambda result: result.metrics['pos_intervention_accs']['y'],
        eval_mode='pos_intervention',
        save_dir=save_dir,
        save_name='pos_intervention',
        prefix=name,
        show=show,
    )

def plot_random_concepts_residual(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
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
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plt.clf()
    save_path = get_save_path(plot_key, prefix=name, suffix='random', save_dir=save_dir)

    baseline_accuracies = []
    random_concept_accuracies = []
    random_residual_accuracies = []

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    info = (
        (baseline_accuracies, "accuracy", "test_acc"),
        (random_concept_accuracies, "random_concepts", "random_concept_acc"),
        (random_residual_accuracies, "random_residual", "random_residual_acc"),
    )
    for key in plot_results.keys():
        results = group_results(plot_results[key], groupby="eval_mode")
        for collection, eval_mode, metric in info:
            collection.append(
                np.mean([result.metrics[metric] for result in results[eval_mode]])
            )

    # Create CSV file
    data = np.stack([
        baseline_accuracies,
        random_concept_accuracies,
        random_residual_accuracies,
    ], axis=1)
    columns = ['Baseline', 'Random Concepts', 'Random Residual']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path.with_suffix('.csv'), index=False)

    # Create figure
    x = np.arange(len(plot_results.keys()))
    plt.bar(x - 0.25, baseline_accuracies, label="Baseline", width=0.25)
    plt.bar(x, random_concept_accuracies, label="Random Concepts", width=0.25)
    plt.bar(x + 0.25, random_residual_accuracies, label="Random Residual", width=0.25)
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Classification Accuracy")
    plt.title(f"Random Concepts & Residual: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path.with_suffix('.png'))
    if show:
        plt.show()

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
    plt.clf()
    save_path = get_save_path(
        plot_key, prefix=name, suffix='disentanglement', save_dir=save_dir)

    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    correlations, mutual_infos = [], []
    for key in plot_results.keys():
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
        correlations.append(correlation)
        mutual_infos.append(mutual_info)
        plt.scatter(correlation, mutual_info, label=key)

    # Create CSV file
    data = np.stack([correlations, mutual_infos], axis=1)
    columns = ['Mean Absolute Cross-Correlation', 'Mutual Information']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path.with_suffix('.csv'), index=False)

    # Create figure
    plt.xlabel("Mean Absolute Cross-Correlation")
    plt.ylabel("Mutual Information")
    plt.title(f"Disentanglement Metrics: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path)
    if show:
        plt.show()

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
        if "pos_intervention" not in results:
            print("No positive intervention results found for:", key)
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
            save_path = get_save_path(
                plot_key,
                prefix=name,
                suffix=f'{x_axis_type}_vs_{y_axis_type}',
                save_dir=save_dir,
            )

            # Create CSV file
            x, y = eval(x_axis_type), eval(y_axis_type)
            data = np.stack([x, y], axis=1)
            columns = [x_axis_type, y_axis_type]
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(save_path.with_suffix('.csv'), index=False)

            # Create figure
            plt.clf()
            plt.scatter(x, y)
            plt.xlabel(f"{x_axis_type}")
            plt.ylabel(f"{y_axis_type}")
            plt.title(f": {format_plot_title((x_axis_type, y_axis_type))} {name}")
            #plt.legend()
            plt.savefig(save_path.with_suffix('.png'))
            if show:
                plt.show()



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
    print("Loading evaluation results from", experiment_path / 'eval')
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
