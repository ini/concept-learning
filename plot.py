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
from ray_utils import group_results



### Typing

Results = Union[Iterable[ray.train.Result], dict[str, 'Results']]



### Helper Functions

def get_dataset_title(dataset_name: str) -> str:
    """
    Get a nicely-formatted title for the given dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    """
    dataset_title = dataset_name.replace('_', ' ').title()
    dataset_title = dataset_title.replace('Mnist', 'MNIST')
    dataset_title = dataset_title.replace('Cifar', 'CIFAR')
    dataset_title = dataset_title.replace('Cub', 'CUB')
    return dataset_title

def load_eval_results(path: str) -> dict[str, ResultGrid]:
    """
    Get evaluation results for the given experiment.

    Parameters
    ----------
    path : str
        Path to the experiment directory

    Returns
    -------
    results : dict
        Evaluation results grouped by dataset
    """
    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(path).resolve().glob('**/eval/tuner.pkl')
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent

    # Load evaluation results
    print('Loading evaluation results from', experiment_path)
    tuner = tune.Tuner.restore(str(experiment_path), trainable=evaluate)
    results = tuner.get_results()
    return group_results(results, groupby='dataset')



### Plotting

def plot_negative_interventions(
    dataset_results: ResultGrid, dataset_name: str, groupby: list[str] = ['model_type']):
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
    """
    groupby = groupby[0] if len(groupby) == 1 else groupby
    for key, results in group_results(dataset_results, groupby=groupby).items():
        results = group_results(results, groupby='eval_mode')
        for result in results.get('neg_intervention', []):
            accuracies = np.array(result.metrics['neg_intervention_accs'])
            plt.plot(range(len(accuracies)), 1 - accuracies, label=key)

    plt.xlabel('# of Concepts Intervened')
    plt.ylabel('Classification Error')
    plt.title(f'Negative Interventions: {get_dataset_title(dataset_name)}')
    plt.legend()
    plt.show()

def plot_positive_interventions(
    dataset_results: ResultGrid, dataset_name: str, groupby: list[str] = ['model_type']):
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
    """
    groupby = groupby[0] if len(groupby) == 1 else groupby
    for key, results in group_results(dataset_results, groupby=groupby).items():
        results = group_results(results, groupby='eval_mode')
        for result in results.get('pos_intervention', []):
            accuracies = np.array(result.metrics['pos_intervention_accs'])
            plt.plot(range(len(accuracies)), accuracies, label=key)

    plt.xlabel('# of Concepts Intervened')
    plt.ylabel('Classification Accuracy')
    plt.title(f'Positive Interventions: {get_dataset_title(dataset_name)}')
    plt.legend()
    plt.show()

def plot_random_concepts_residual(
    dataset_results: ResultGrid, dataset_name: str, groupby: list[str] = ['model_type']):
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
    """
    baseline_accuracies = []
    random_concept_accuracies = []
    random_residual_accuracies = []

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    dataset_results = group_results(dataset_results, groupby=groupby)
    keys = sorted(dataset_results.keys())
    for key in keys:
        results = group_results(dataset_results[key], groupby='eval_mode')
        baseline_accuracies.append(
            results['accuracy'][0].metrics['test_acc'])
        random_concept_accuracies.append(
            results['random_concepts'][0].metrics['random_concept_acc'])
        random_residual_accuracies.append(
            results['random_residual'][0].metrics['random_residual_acc'])

    # Plot
    plt.bar(
        np.arange(len(keys)) - 0.25,
        baseline_accuracies,
        label='Baseline',
        width=0.25,
    )
    plt.bar(
        np.arange(len(keys)),
        random_concept_accuracies,
        label='Random Concepts',
        width=0.25,
    )
    plt.bar(
        np.arange(len(keys)) + 0.25,
        random_residual_accuracies,
        width=0.25,
        label='Random Residual',
    )

    y_min = min([
        *baseline_accuracies,
        *random_concept_accuracies,
        *random_residual_accuracies,
    ])
    plt.xticks(np.arange(len(keys)), keys)
    plt.ylim(max(0, y_min - 0.1), 1)
    plt.ylabel('Classification Accuracy')
    plt.title(f'Random Concepts & Residual: {get_dataset_title(dataset_name)}')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    MODES = ['neg_intervention', 'pos_intervention', 'random']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-dir', type=str, default=os.environ.get('CONCEPT_SAVE_DIR', './saved'),
        help='Experiment directory')
    parser.add_argument(
        '--mode', nargs='+', default=MODES, help='Evaluation modes')
    parser.add_argument(
        '--groupby', nargs='+', default=['model_type'],
        help='Train config keys to group plots by'
    )

    args = parser.parse_args()
    eval_results = load_eval_results(args.exp_dir)

    # Plot results for each dataset
    for dataset_name, dataset_results in eval_results.items():
        for mode in args.mode:
            if mode == 'neg_intervention':
                plot_negative_interventions(
                    dataset_results, dataset_name, args.groupby)
            elif mode == 'pos_intervention':
                plot_positive_interventions(
                    dataset_results, dataset_name, args.groupby)
            elif mode == 'random':
                plot_random_concepts_residual(
                    dataset_results, dataset_name, args.groupby)
