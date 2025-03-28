from __future__ import annotations

import argparse
from functools import partial
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
    items = [str(key).replace(".", "_") for key in plot_key]
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


# def plot_concept_predictions(
#     plot_results: ResultGrid,
#     plot_key: tuple[str, ...],
#     groupby: list[str] = ["model_type"],
#     save_dir: Path | str = "./plots",
#     show: bool = True,
#     name: str = "",
# ):
#     """
#     Plot positive intervention results.

#     Parameters
#     ----------
#     plot_results : ResultGrid
#         Results for the given plot
#     plot_key : tuple[str]
#         Identifier for this plot
#     groupby : list[str]
#         List of train config keys to group by
#     save_dir : Path or str
#         Directory to save plots to
#     show : bool
#         Whether to show the plot
#     """
#     plot_curves(
#         plot_results,
#         plot_key,
#         groupby=groupby,
#         title=f"Concept Predictions: {format_plot_title(plot_key)} {name}",
#         x_label="Concept #",
#         y_label="Concept Prediction Accuracy",
#         get_x=lambda results: np.linspace(
#             0, 1, len(results[0].metrics["concept_pred"])
#         ),
#         get_y=lambda result: result.metrics["concept_pred"],
#         eval_mode="concept_pred",
#         save_dir=save_dir,
#         save_name="concept_pred",
#         prefix=name,
#         show=show,
#     )


def plot_concept_predictions(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_hidden_concepts: bool = False,
):
    """
    Plot results for concept disentanglement.

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
    save_path_disentanglement = get_save_path(
        plot_key, prefix=name, suffix="disentanglement", save_dir=save_dir
    )
    save_path_change = get_save_path(
        plot_key, prefix=name, suffix="change", save_dir=save_dir
    )

    supervised_accs, hidden_accs = [], []
    supervised_change, hidden_change = [], []
    supervised_accs_std, hidden_accs_std = [], []
    supervised_change_std, hidden_change_std = [], []

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    info = (
        (supervised_accs, supervised_accs_std, "concept_pred", 0),
        (supervised_change, supervised_change_std, "concept_pred", 2),
    )
    if plot_hidden_concepts:
        info += (
            (hidden_accs, hidden_accs_std, "concept_pred", 1),
            (hidden_change, hidden_change_std, "concept_pred", 3),
        )
    for key in plot_results.keys():
        results = group_results(plot_results[key], groupby="eval_mode")
        for values, stds, eval_mode, metric_idx in info:
            values.append(
                np.mean(
                    [
                        result.metrics[eval_mode][metric_idx]
                        for result in results[eval_mode]
                    ]
                )
            )
            stds.append(
                np.std(
                    [
                        result.metrics[eval_mode][metric_idx]
                        for result in results[eval_mode]
                    ]
                )
            )

    # Create CSV file
    if plot_hidden_concepts:
        data = np.stack(
            [
                supervised_accs,
                hidden_accs,
                supervised_change,
                hidden_change,
            ],
            axis=1,
        )
        columns = [
            "Supervised Concepts Acc.",
            "Hidden Concepts Acc.",
            "Supervised Concepts Change",
            "Hidden Concepts Change",
        ]
    else:
        data = np.stack(
            [
                supervised_accs,
                supervised_change,
            ],
            axis=1,
        )
        columns = [
            "Supervised Concepts Acc.",
            "Supervised Concepts Change",
        ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path_disentanglement.with_suffix(".csv"), index=False)

    # Create figure for concept accuracy
    x = np.arange(len(plot_results.keys()))
    plt.figure()
    plt.bar(
        x - 0.2,
        supervised_accs,
        yerr=supervised_accs_std,
        label="Supervised Concepts Acc.",
        width=0.2,
        capsize=5,
    )
    if plot_hidden_concepts:
        plt.bar(
            x,
            hidden_accs,
            yerr=hidden_accs_std,
            label="Hidden Concepts Acc.",
            width=0.2,
            capsize=5,
        )
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Accuracy")
    plt.title(f"Concept Accuracy: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path_disentanglement.with_suffix(".png"))
    if show:
        plt.show()

    # Create figure for concept prediction change
    plt.figure()
    plt.bar(
        x - 0.2,
        supervised_change,
        yerr=supervised_change_std,
        label="Supervised Concepts Change",
        width=0.2,
        capsize=5,
    )
    if plot_hidden_concepts:
        plt.bar(
            x,
            hidden_change,
            yerr=hidden_change_std,
            label="Hidden Concepts Change",
            width=0.2,
            capsize=5,
        )
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Change")
    plt.title(f"Concept Prediction Change: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path_change.with_suffix(".png"))
    if show:
        plt.show()


def plot_concept_change(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_hidden_concepts: bool = False,
):
    """
    Plot results for concept change.

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
    plot_hidden_concepts : bool
        Whether to plot hidden concepts
    """
    plt.clf()
    save_path_change = get_save_path(
        plot_key, prefix=name, suffix="concept_change", save_dir=save_dir
    )

    num_changed_concepts, concept_updated_when_wrong, hidden_concepts_updated = (
        [],
        [],
        [],
    )
    (
        num_changed_concepts_std,
        concept_updated_when_wrong_std,
        hidden_concepts_updated_std,
    ) = (
        [],
        [],
        [],
    )

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    for key in plot_results.keys():
        results = plot_results[key]
        num_changed_concepts.append(
            np.mean([result.metrics["concept_change"][0] for result in results])
        )
        concept_updated_when_wrong.append(
            np.mean([result.metrics["concept_change"][1] for result in results])
        )
        hidden_concepts_updated.append(
            np.mean([result.metrics["concept_change"][2] for result in results])
        )
        num_changed_concepts_std.append(
            np.std([result.metrics["concept_change"][0] for result in results])
        )
        concept_updated_when_wrong_std.append(
            np.std([result.metrics["concept_change"][1] for result in results])
        )
        hidden_concepts_updated_std.append(
            np.std([result.metrics["concept_change"][2] for result in results])
        )

    # Create CSV file
    data = np.stack(
        [
            num_changed_concepts,
            concept_updated_when_wrong,
            hidden_concepts_updated,
            num_changed_concepts_std,
            concept_updated_when_wrong_std,
            hidden_concepts_updated_std,
        ],
        axis=1,
    )
    columns = [
        "Num Changed Concepts",
        "Concept Updated When Wrong",
        "Hidden Concepts Updated",
        "Num Changed Concepts Std.",
        "Concept Updated When Wrong Std.",
        "Hidden Concepts Updated Std.",
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path_change.with_suffix(".csv"), index=False)

    # Create figure for concept change
    x = np.arange(len(plot_results.keys()))
    plt.figure()
    plt.bar(
        x - 0.2,
        num_changed_concepts,
        yerr=num_changed_concepts_std,
        label="Num Changed Concepts",
        width=0.2,
        capsize=5,
    )
    plt.bar(
        x,
        concept_updated_when_wrong,
        yerr=concept_updated_when_wrong_std,
        label="Concept Updated",
        width=0.2,
        capsize=5,
    )
    if plot_hidden_concepts:
        plt.bar(
            x + 0.2,
            hidden_concepts_updated,
            yerr=hidden_concepts_updated_std,
            label="Hidden Concepts Updated",
            width=0.2,
            capsize=5,
        )
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Metrics")
    plt.title(f"Concept Change: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path_change.with_suffix(".png"))
    if show:
        plt.show()


def plot_concept_change_probe(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_hidden_concepts: bool = False,
):
    """
    Plot results for concept change probe.

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
    plot_hidden_concepts : bool
        Whether to plot hidden concepts
    """
    plt.clf()
    save_path_change_probe = get_save_path(
        plot_key, prefix=name, suffix="concept_change_probe", save_dir=save_dir
    )

    accuracy, num_changed_concepts, concept_updated, hidden_concepts_updated = (
        [],
        [],
        [],
        [],
    )
    (
        accuracy_std,
        num_changed_concepts_std,
        concept_updated_std,
        hidden_concepts_updated_std,
    ) = ([], [], [], [])

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    for key in plot_results.keys():
        results = plot_results[key]
        accuracy.append(
            np.mean([result.metrics["concept_change_probe"][0] for result in results])
        )
        num_changed_concepts.append(
            np.mean([result.metrics["concept_change_probe"][1] for result in results])
        )
        concept_updated.append(
            np.mean([result.metrics["concept_change_probe"][2] for result in results])
        )
        hidden_concepts_updated.append(
            np.mean([result.metrics["concept_change_probe"][3] for result in results])
        )
        accuracy_std.append(
            np.std([result.metrics["concept_change_probe"][0] for result in results])
        )
        num_changed_concepts_std.append(
            np.std([result.metrics["concept_change_probe"][1] for result in results])
        )
        concept_updated_std.append(
            np.std([result.metrics["concept_change_probe"][2] for result in results])
        )
        hidden_concepts_updated_std.append(
            np.std([result.metrics["concept_change_probe"][3] for result in results])
        )

    # Create CSV file
    data = np.stack(
        [
            num_changed_concepts,
            concept_updated,
            hidden_concepts_updated,
            num_changed_concepts_std,
            concept_updated_std,
            hidden_concepts_updated_std,
        ],
        axis=1,
    )
    columns = [
        "Accuracy",
        "Num Changed Concepts",
        "Concept Updated",
        "Hidden Concepts Updated",
        "Num Changed Concepts Std.",
        "Concept Updated Std.",
        "Hidden Concepts Updated Std.",
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path_change_probe.with_suffix(".csv"), index=False)

    # Create figure for concept change probe
    x = np.arange(len(plot_results.keys()))
    plt.figure()
    plt.bar(
        x - 0.3,
        accuracy,
        yerr=accuracy_std,
        label="Accuracy",
        width=0.2,
        capsize=5,
    )
    plt.bar(
        x - 0.1,
        num_changed_concepts,
        yerr=num_changed_concepts_std,
        label="Num Changed Concepts",
        width=0.2,
        capsize=5,
    )
    plt.bar(
        x + 0.1,
        concept_updated,
        yerr=concept_updated_std,
        label="Concept Updated",
        width=0.2,
        capsize=5,
    )
    if plot_hidden_concepts:
        plt.bar(
            x + 0.3,
            hidden_concepts_updated,
            yerr=hidden_concepts_updated_std,
            label="Hidden Concepts Updated",
            width=0.2,
            capsize=5,
        )
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Metrics")
    plt.title(f"Concept Change Probe: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path_change_probe.with_suffix(".png"))
    if show:
        plt.show()


def plot_tcav(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_magnitude: bool = False,
):
    """
    Plot results for concept change.

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
    plot_hidden_concepts : bool
        Whether to plot hidden concepts
    """
    plt.clf()
    pm = "_magnitude" if plot_magnitude else ""
    save_path = get_save_path(
        plot_key, prefix=name, suffix=f"tcav{pm}", save_dir=save_dir
    )

    def tcav_process(result, subkey):
        """
        Extracts the TCAV scores from the result.
        """
        if plot_magnitude:
            # Use magnitude instead of sign_count for the plot
            metric_to_use = "magnitude"
        else:
            metric_to_use = "sign_count"
        # metric_to_use = "magnitude"
        return {
            key: (
                np.mean(value[metric_to_use][subkey]),
                np.std(value[metric_to_use][subkey])
                / np.sqrt(len(value[metric_to_use][subkey])),
            )
            for key, value in result.metrics["tcav_scores"].items()
        }

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    all_scores = {}
    for key in plot_results.keys():
        results = plot_results[key]
        tcav_scores = [
            tcav_process(result, "P1")
            for result in results
            if "tcav_scores" in result.metrics
        ]
        all_scores[key] = tcav_scores

    all_concepts = set()
    for method, scores_list in all_scores.items():
        for scores in scores_list:
            all_concepts.update(scores.keys())

    all_concepts = sorted(list(all_concepts))
    methods = sorted(list(all_scores.keys()))

    # Create figure with one subplot per concept
    # n_concepts = len(all_concepts)
    # fig, axes = plt.subplots(1, n_concepts, figsize=(5 * n_concepts, 8), sharey=True)

    # # If there's only one concept, axes won't be an array
    # if n_concepts == 1:
    #     axes = [axes]

    # # Plot each concept
    # for i, concept in enumerate(all_concepts):
    #     ax = axes[i]

    #     # Set up bar positions with spacing for each method
    #     bar_width = 0.8 / len(methods)
    #     method_positions = np.arange(len(methods))

    #     # Plot each method
    #     for j, method in enumerate(methods):
    #         means = []
    #         stds = []
    #         for score_dict in all_scores[method]:
    #             if concept in score_dict:
    #                 mean, std = score_dict[concept]
    #                 print(concept, score_dict[concept])
    #                 means.append(mean)
    #                 stds.append(std)

    #         if means:
    #             # Calculate the position for each bar with appropriate spacing
    #             x_positions = method_positions[j]

    #             # Plot mean as a ba
    #             print(stds)
    #             ax.bar(
    #                 x_positions,
    #                 np.mean(means),  # np.abs(0.5 - np.mean(means)),
    #                 width=bar_width,
    #                 yerr=np.mean(stds),  # Standard error
    #                 capsize=5,
    #                 color=plt.cm.tab10.colors[j],
    #                 edgecolor="black",
    #                 alpha=0.7,
    #                 label=str(method),
    #             )

    #             # Add individual data points if there are multiple results
    #             if len(means) > 1:
    #                 scatter_positions = np.random.normal(
    #                     x_positions, bar_width / 8, len(means)
    #                 )
    #                 ax.scatter(scatter_positions, means, color="black", s=20, zorder=3)

    #     # Set title and labels
    #     ax.set_title(f"{concept}", fontsize=14)
    #     if i == 0:
    #         ax.set_ylabel("TCAV Score")
    #     ax.grid(axis="y", linestyle="--", alpha=0.3)

    #     # Set x-axis labels
    #     ax.set_xticks(method_positions)
    #     ax.set_xticklabels(methods, rotation=45, ha="right")

    # # Add legend in the last subplot only if there are multiple methods
    # if len(methods) > 1:
    #     handles, labels = axes[-1].get_legend_handles_labels()
    #     if handles:
    #         fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    # plt.tight_layout()
    # plt.suptitle("TCAV Scores Across Methods", fontsize=16, y=1.02)
    # plt.savefig(save_path.with_suffix(".png"))

    # if show:
    #     plt.show()
    n_concepts = len(all_concepts)

    # Calculate grid dimensions - max 4 concepts per row
    max_cols = 4
    n_cols = min(n_concepts, max_cols)
    n_rows = (n_concepts + max_cols - 1) // max_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=False
    )

    # If there's only one row, axes might not be a 2D array
    if n_rows == 1:
        if n_cols == 1:
            axes = np.array([[axes]])
        else:
            axes = np.array([axes])

    # Plot each concept
    for i, concept in enumerate(all_concepts):
        # Calculate row and column for this concept
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Set up bar positions with spacing for each method
        bar_width = 0.8 / len(methods)
        method_positions = np.arange(len(methods))

        # Plot each method
        for j, method in enumerate(methods):
            means = []
            stds = []
            for score_dict in all_scores[method]:
                if concept in score_dict:
                    mean, std = score_dict[concept]
                    print(concept, score_dict[concept])
                    means.append(mean)
                    stds.append(std)

            if means:
                # Calculate the position for each bar with appropriate spacing
                x_positions = method_positions[j]

                # Plot mean as a bar
                print(stds)
                ax.bar(
                    x_positions,
                    np.mean(means),  # np.abs(0.5 - np.mean(means)),
                    width=bar_width,
                    yerr=np.mean(stds),
                    capsize=5,
                    color=plt.cm.tab10.colors[
                        j % 10
                    ],  # Cycle through colors if more than 10 methods
                    edgecolor="black",
                    alpha=0.7,
                    label=str(method),
                )

                # Add individual data points if there are multiple results
                if len(means) > 1:
                    scatter_positions = np.random.normal(
                        x_positions, bar_width / 8, len(means)
                    )
                    ax.scatter(scatter_positions, means, color="black", s=20, zorder=3)
        all_values = []
        for m_idx, m_name in enumerate(methods):
            for score_dict in all_scores[m_name]:
                if concept in score_dict:
                    mean, std = score_dict[concept]
                    all_values.append(mean)
                    # Include error bar values
                    all_values.append(mean + std)
                    all_values.append(
                        max(0, mean - std)
                    )  # Prevent negative values if not meaningful

        if all_values:
            # Set y-axis limits with 10% padding
            min_val = min(all_values)
            max_val = max(all_values)
            y_range = max_val - min_val

            # If range is very small, use a minimum range to avoid tiny plots
            if y_range < 0.05:
                padding = 0.05
            else:
                padding = y_range * 0.15  # 15% padding

            # Set limits, ensuring we don't go below 0 if values are all positive
            y_min = max(0, min_val - padding) if min_val >= 0 else min_val - padding
            y_max = max_val + padding

            ax.set_ylim(y_min, y_max)
        # Set title and labels
        ax.set_title(f"{concept}", fontsize=14)
        if col == 0:  # Only set y-label for leftmost plots
            ax.set_ylabel("TCAV Score")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Set x-axis labels
        ax.set_xticks(method_positions)
        ax.set_xticklabels(methods, rotation=45, ha="right")

    # Hide unused subplots
    for i in range(n_concepts, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    # Add legend only once for the entire figure
    if len(methods) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    plt.suptitle("TCAV Scores Across Methods", fontsize=16, y=1.02)
    plt.savefig(save_path.with_suffix(".png"))

    if show:
        plt.show()


if __name__ == "__main__":
    PLOT_FUNCTIONS = {
        "neg_intervention": plot_negative_interventions,
        "pos_intervention": plot_positive_interventions,
        "random": plot_random_concepts_residual,
        # "concept_pred": plot_concept_predictions,
        # "concept_change": plot_concept_change,
        # "concept_change_probe": plot_concept_change_probe,
        # "concept_change": plot_concept_changes,
        # "disentanglement": plot_disentanglement,
        # "intervention_vs_disentanglement": plot_intervention_vs_disentanglement,
        "tcav_sign_count": partial(plot_tcav, plot_magnitude=False),
        "tcav_magnitude": partial(plot_tcav, plot_magnitude=True),
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

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    folder = "eval"
    experiment_paths = Path(args.exp_dir).resolve().glob(f"**/{folder}/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load evaluation results
    print("Loading evaluation results from", experiment_path / folder)
    tuner = tune.Tuner.restore(str(experiment_path / folder), trainable=evaluate)
    results = group_results(tuner.get_results(), groupby=args.plotby)

    # Plot results
    plot_folder = "plots"
    for plot_key, plot_results in tqdm(results.items()):
        for mode in tqdm(args.mode):
            print(plot_key)
            PLOT_FUNCTIONS[mode](
                plot_results,
                plot_key,
                groupby=args.groupby,
                save_dir=experiment_path / plot_folder,
                show=args.show,
            )
