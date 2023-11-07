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
from lightning_ray import group_results



### Typing

Results = Union[Iterable[ray.train.Result], dict[str, "Results"]]



### Helper Functions

def get_dataset_title(dataset_name: str) -> str:
    """
    Get a nicely-formatted title for the given dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    """
    dataset_title = dataset_name.replace("_", " ").title()
    dataset_title = dataset_title.replace("Mnist", "MNIST")
    dataset_title = dataset_title.replace("Cifar", "CIFAR")
    dataset_title = dataset_title.replace("Cub", "CUB")
    return dataset_title



### Plotting

def plot_negative_interventions(
    dataset_results: ResultGrid,
    dataset_name: str,
    groupby: list[str] = ['model_type'],
    save_dir: Path | str = './plots',
    show: bool = True,
):
    """
    Plot negative intervention results.

    Parameters
    ----------
    dataset_results : ResultGrid
        Results for the given dataset, grouped by model type and evaluation mode
    dataset_name : str
        Name of the dataset
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
    for key, results in group_results(dataset_results, groupby=groupby).items():
        results = group_results(results, groupby="eval_mode")
        for result in results.get('neg_intervention', []):
            num_interventions = result.metrics['neg_intervention_accs']['x']
            accuracies = result.metrics['neg_intervention_accs']['y']
            plt.plot(num_interventions, 1 - accuracies, label=key)

    plt.xlabel("# of Concepts Intervened")
    plt.ylabel("Classification Error")
    plt.title(f"Negative Interventions: {get_dataset_title(dataset_name)}")
    plt.legend()
    plt.savefig(save_dir / f"{dataset_name}_neg_intervention.png")
    if show:
        plt.show()

def plot_positive_interventions(
    dataset_results: ResultGrid,
    dataset_name: str,
    groupby: list[str] = ['model_type'],
    save_dir: Path | str = './plots',
    show: bool = True,
):
    """
    Plot negative intervention results.

    Parameters
    ----------
    dataset_results : ResultGrid
        Results for the given dataset, grouped by model type and evaluation mode
    dataset_name : str
        Name of the dataset
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
    for key, results in group_results(dataset_results, groupby=groupby).items():
        results = group_results(results, groupby='eval_mode')
        for result in results.get('pos_intervention', []):
            num_interventions = result.metrics['pos_intervention_accs']['x']
            accuracies = result.metrics['pos_intervention_accs']['y']
            plt.plot(num_interventions, accuracies, label=key)

    plt.xlabel("# of Concepts Intervened")
    plt.ylabel("Classification Accuracy")
    plt.title(f"Positive Interventions: {get_dataset_title(dataset_name)}")
    plt.legend()
    plt.savefig(save_dir / f"{dataset_name}_pos_intervention.png")
    if show:
        plt.show()

def plot_random_concepts_residual(
    dataset_results: ResultGrid,
    dataset_name: str,
    groupby: list[str] = ['model_type'],
    save_dir: Path | str = './plots',
    show: bool = True,
):
    """
    Plot results with randomized concepts and residuals.

    Parameters
    ----------
    dataset_results : ResultGrid
        Results for the given dataset
    dataset_name : str
        Name of the dataset
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
    dataset_results = group_results(dataset_results, groupby=groupby)
    keys = sorted(dataset_results.keys())
    info = (
        (baseline_accuracies, 'accuracy', 'test_acc'),
        (random_concept_accuracies, 'random_concepts', 'random_concept_acc'),
        (random_residual_accuracies, 'random_residual', 'random_residual_acc'),
    )
    for key in keys:
        results = group_results(dataset_results[key], groupby='eval_mode')
        for collection, eval_mode, metric in info:
            collection.append(
                np.mean([
                    result.metrics[metric]
                    for result in results[eval_mode]
                ])
            )

    # Plot
    x = np.arange(len(keys))
    plt.bar(x - 0.25, baseline_accuracies, label='Baseline', width=0.25)
    plt.bar(x, random_concept_accuracies, label='Random Concepts', width=0.25)
    plt.bar(x + 0.25, random_residual_accuracies, label='Random Residual', width=0.25)

    y_min = min([
        *baseline_accuracies,
        *random_concept_accuracies,
        *random_residual_accuracies,
    ])
    plt.xticks(np.arange(len(keys)), keys)
    plt.ylim(max(0, y_min - 0.1), 1)
    plt.ylabel("Classification Accuracy")
    plt.title(f"Random Concepts & Residual: {get_dataset_title(dataset_name)}")
    plt.legend()
    plt.savefig(save_dir / f"{dataset_name}_random.png")
    if show:
        plt.show()

def plot_disentanglement(
    dataset_results: ResultGrid,
    dataset_name: str,
    groupby: list[str] = ['model_type'],
    save_dir: Path | str = './plots',
    show: bool = True,
):
    """
    Plot disentanglement metrics.

    Parameters
    ----------
    dataset_results : ResultGrid
        Results for the given dataset
    dataset_name : str
        Name of the dataset
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
    dataset_results = group_results(dataset_results, groupby=groupby)
    keys = sorted(dataset_results.keys())
    for key in keys:
        results = group_results(dataset_results[key], groupby='eval_mode')
        correlation = np.mean([
            result.metrics['mean_abs_cross_correlation']
            for result in results['correlation']
        ])
        mutual_info = np.mean([
            result.metrics['mutual_info']
            for result in results['mutual_info']
        ])
        plt.scatter(correlation, mutual_info, label=key)

    plt.xlabel("Mean Absolute Cross-Correlation")
    plt.ylabel("Mutual Information")
    plt.title(f"Disentanglement Metrics: {get_dataset_title(dataset_name)}")
    plt.legend()
    plt.savefig(save_dir / f"{dataset_name}_disentanglement.png")
    if show:
        plt.show()



if __name__ == '__main__':
    PLOT_FUNCTIONS = {
        'neg_intervention': plot_negative_interventions,
        'pos_intervention': plot_positive_interventions,
        'random': plot_random_concepts_residual,
        'disentanglement': plot_disentanglement,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-dir',
        type=str,
        default=os.environ.get('CONCEPT_SAVE_DIR', './saved'),
        help="Experiment directory",
    )
    parser.add_argument(
        '--mode',
        nargs='+',
        default=PLOT_FUNCTIONS.keys(),
        help="Plot modes",
    )
    parser.add_argument(
        '--plotby',
        nargs='+',
        default=['dataset'],
        help=(
            "Config keys to group plots by "
            "(e.g. `--plotby dataset model_type` creates separate plots "
            "for each (dataset, model_type) combination)"
        )
    )
    parser.add_argument(
        '--groupby',
        nargs='+',
        default=['model_type'],
        help="Config keys to group results on each plot by",
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob('**/eval/tuner.pkl')
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load evaluation results
    tuner = tune.Tuner.restore(str(experiment_path / 'eval'), trainable=evaluate)
    results = group_results(tuner.get_results(), groupby=args.plotby)

    # Plot results
    for plot_key, plot_results in results.items():
        for mode in args.mode:
            PLOT_FUNCTIONS[mode](
                plot_results,
                plot_key,
                groupby=args.groupby,
                save_dir=experiment_path / 'plots',
            )
