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
from tqdm import tqdm
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
            return ", ".join([format_plot_title(key) for key in plot_key])
        else:
            plot_key = plot_key[0]

    if isinstance(plot_key, str):
        plot_key = plot_key.replace("_", " ").title()
        plot_key = plot_key.replace("Mnist", "MNIST")
        plot_key = plot_key.replace("Cifar", "CIFAR")
        plot_key = plot_key.replace("Cub", "CUB")
        plot_key = plot_key.replace("Oai", "OAI")
        plot_key = plot_key.replace("Mi Residual", "Mutual Info Residual")
        plot_key = plot_key.replace("Iter Norm", "IterNorm Residual")

    return str(plot_key)


def get_save_path(
    plot_key: tuple,
    prefix: str | None = None,
    suffix: str | None = None,
    save_dir: Path | str = "./plots",
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
    return save_dir / "_".join(items)


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
    """
    Create a plot with curve(s) for the specified results.
    """
    plt.clf()
    save_path = get_save_path(
        plot_key, prefix=prefix, suffix=save_name, save_dir=save_dir
    )

    data, columns = [], []
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    for key, results in plot_results.items():
        results = group_results(results, groupby="eval_mode")
        if eval_mode not in results:
            tqdm.write(f"No {eval_mode} results found for: {plot_key} {groupby} {key}")
            continue

        x = get_x(results[eval_mode])
        y = np.stack([get_y(result) for result in results[eval_mode]]).mean(axis=0)
        y_std = np.stack([get_y(result) for result in results[eval_mode]]).std(axis=0)
        plt.plot(x, y, label=key)
        data.extend([y, y_std])
        columns.extend([f"{key} {y_label}", f"{key} {y_label} Std."])

    # Create CSV file
    x = np.linspace(0, 1, len(data[0]))
    data = np.stack([x, *data], axis=1)
    columns.insert(0, x_label)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Create figure
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path.with_suffix(".png"))
    if show:
        plt.show()


def plot_scatter(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str],
    title: str,
    x_label: str,
    y_label: str,
    x_eval_mode: str,
    y_eval_mode: str,
    get_x: Callable[[ResultGrid], float | np.ndarray],
    get_y: Callable[[ResultGrid], float | np.ndarray],
    save_dir: Path | str,
    save_name: str,
    prefix: str | None = None,
    show_regression_line: bool = False,
    show: bool = True,
):
    """
    Create a scatter plot for the specified results.
    """
    plt.clf()
    save_path = get_save_path(
        plot_key, prefix=prefix, suffix=save_name, save_dir=save_dir
    )

    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    x_values, y_values, index = [], [], []
    for key in plot_results.keys():
        results = group_results(plot_results[key], groupby="eval_mode")
        if x_eval_mode not in results or np.isnan(get_x(results[x_eval_mode])).any():
            tqdm.write(
                f"No {x_eval_mode} results found for: {plot_key} {groupby} {key}"
            )
            continue
        if y_eval_mode not in results or np.isnan(get_y(results[y_eval_mode])).any():
            tqdm.write(
                f"No {y_eval_mode} results found for: {plot_key} {groupby} {key}"
            )
            continue

        x = get_x(results[x_eval_mode])
        y = get_y(results[y_eval_mode])
        plt.scatter(x, y, label=key)
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_values.extend(x.flatten())
            y_values.extend(y.flatten())
            index += [key] * len(x)
        else:
            x_values.append(x)
            y_values.append(y)
            index.append(key)

    if len(x_values) == 0 or len(y_values) == 0:
        tqdm.write(f"No {save_name} results found for: {plot_key}")
        return

    # Create CSV file
    data = np.stack([x_values, y_values], axis=1)
    df = pd.DataFrame(data, index=index, columns=[x_label, y_label])
    df.to_csv(save_path.with_suffix(".csv"), index=True)

    # Add linear regression line
    if show_regression_line:
        from scipy.stats import linregress

        x, y = np.array(x_values), np.array(y_values)
        m, b, r, _, _ = linregress(x, y)
        plt.plot(x, m * x + b, color="black")
        plt.text(
            x=0.5,
            y=0.5,
            s=f"$r^2$ = {r**2:.3f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.75),
        )

    # Create figure
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path.with_suffix(".png"))
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
            0, 1, len(results[0].metrics["neg_intervention_accs"]["y"])
        ),
        get_y=lambda result: 1 - result.metrics["neg_intervention_accs"]["y"],
        eval_mode="neg_intervention",
        save_dir=save_dir,
        save_name="neg_intervention",
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
            0, 1, len(results[0].metrics["pos_intervention_accs"]["y"])
        ),
        get_y=lambda result: result.metrics["pos_intervention_accs"]["y"],
        eval_mode="pos_intervention",
        save_dir=save_dir,
        save_name="pos_intervention",
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
    save_path = get_save_path(plot_key, prefix=name, suffix="random", save_dir=save_dir)

    baseline_accs, baseline_stds = [], []
    random_concept_accs, random_concept_stds = [], []
    random_residual_accs, random_residual_stds = [], []

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    info = (
        (baseline_accs, baseline_stds, "accuracy", "test_acc"),
        (
            random_concept_accs,
            random_concept_stds,
            "random_concepts",
            "random_concept_acc",
        ),
        (
            random_residual_accs,
            random_residual_stds,
            "random_residual",
            "random_residual_acc",
        ),
    )
    for key in plot_results.keys():
        results = group_results(plot_results[key], groupby="eval_mode")
        for values, stds, eval_mode, metric in info:
            values.append(
                np.mean([result.metrics[metric] for result in results[eval_mode]])
            )
            stds.append(
                np.std([result.metrics[metric] for result in results[eval_mode]])
            )

    # Create CSV file
    data = np.stack(
        [
            baseline_accs,
            baseline_stds,
            random_concept_accs,
            random_concept_stds,
            random_residual_accs,
            random_residual_stds,
        ],
        axis=1,
    )
    columns = [
        "Baseline Acc.",
        "Baseline Std.",
        "Random Concepts Acc.",
        "Random Concepts Std.",
        "Random Residual Acc.",
        "Random Residual Std.",
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Create figure
    x = np.arange(len(plot_results.keys()))
    plt.bar(x - 0.25, baseline_accs, label="Baseline", width=0.25)
    plt.bar(x, random_concept_accs, label="Random Concepts", width=0.25)
    plt.bar(x + 0.25, random_residual_accs, label="Random Residual", width=0.25)
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Classification Accuracy")
    plt.title(f"Random Concepts & Residual: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path.with_suffix(".png"))
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
    x_metric, y_metric = "mean_abs_cross_correlation", "mutual_info"
    plot_scatter(
        plot_results,
        plot_key,
        groupby=groupby,
        title=f"Disentanglement Metrics: {format_plot_title(plot_key)} {name}",
        x_label="Mean Absolute Cross Correlation",
        y_label="Mutual Information",
        x_eval_mode="correlation",
        y_eval_mode="mutual_info",
        get_x=lambda results: np.mean([result.metrics[x_metric] for result in results]),
        get_y=lambda results: np.mean([result.metrics[y_metric] for result in results]),
        save_dir=save_dir,
        save_name="disentanglement",
        prefix=name,
        show_regression_line=False,
        show=show,
    )


def plot_intervention_vs_disentanglement(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot intervention accuracy metrics vs disentanglement metrics.

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
    x_info = [
        (
            "Mean Absolute Cross Correlation",
            "correlation",
            "mean_abs_cross_correlation",
        ),
        ("Mutual Information", "mutual_info", "mutual_info"),
    ]
    y_info = [
        ("Positive Intervention Accuracy", "pos_intervention", "pos_intervention_accs"),
    ]
    for x_label, x_eval_mode, x_metric in x_info:
        for y_label, y_eval_mode, y_metric in y_info:
            if y_metric == "pos_intervention_accs":
                get_y = lambda results: np.stack(
                    [result.metrics[y_metric]["y"][-1] for result in results]
                )
            else:
                get_y = lambda results: np.stack(
                    [result.metrics[y_metric] for result in results]
                )
            plot_scatter(
                plot_results,
                plot_key,
                groupby=groupby,
                title=f"{y_label} vs. {x_label}\n{format_plot_title(plot_key)} {name}",
                x_label=x_label,
                y_label=y_label,
                x_eval_mode=x_eval_mode,
                y_eval_mode=y_eval_mode,
                get_x=lambda results: np.stack(
                    [result.metrics[x_metric] for result in results]
                ),
                get_y=get_y,
                save_dir=save_dir,
                save_name=f"{y_metric}_vs_{x_metric}",
                prefix=name,
                show_regression_line=True,
                show=show,
            )


if __name__ == "__main__":
    PLOT_FUNCTIONS = {
        "neg_intervention": plot_negative_interventions,
        "pos_intervention": plot_positive_interventions,
        "random": plot_random_concepts_residual,
        "disentanglement": plot_disentanglement,
        "intervention_vs_disentanglement": plot_intervention_vs_disentanglement,
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
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether to show the plot(s)",
    )
    parser.add_argument(
        "--evaluate-mixer", action="store_true", help="Evaluate mixer models"
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    folder = "eval" if not args.evaluate_mixer else "eval_mixer"
    experiment_paths = Path(args.exp_dir).resolve().glob(f"**/{folder}/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load evaluation results
    print("Loading evaluation results from", experiment_path / folder)
    tuner = tune.Tuner.restore(str(experiment_path / folder), trainable=evaluate)
    results = group_results(tuner.get_results(), groupby=args.plotby)

    # Plot results
    plot_folder = "plots" if not args.evaluate_mixer else "plots_mixer"
    for plot_key, plot_results in tqdm(results.items()):
        for mode in tqdm(args.mode):
            PLOT_FUNCTIONS[mode](
                plot_results,
                plot_key,
                groupby=args.groupby,
                save_dir=experiment_path / plot_folder,
                show=args.show,
            )
