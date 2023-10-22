from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import ray

from collections import defaultdict
from pathlib import Path
from ray import tune
from typing import Callable, Iterable, Union

from evaluate import evaluate



### Typing

Results = Union[Iterable[ray.train.Result], dict[str, 'Results']]



### Helper Functions

def get_train_config(result: ray.train.Result):
    """
    Get the train config for the given result.

    Parameters
    ----------
    result : ray.train.Result
        Evaluation result
    """
    return result.config['train_result'].config['train_loop_config']

def get_group_key(result: ray.train.Result, groupby: list[str]):
    """
    Get the group key for the given result.

    Parameters
    ----------
    result : ray.train.Result
        Evaluation result
    groupby : list[str]
        List of train config keys to group by
    """
    group_key = tuple(get_train_config(result)[key] for key in groupby)
    return group_key[0] if len(group_key) == 1 else group_key

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

def group_results(results: Results, get_group: Callable) -> dict[str, Results]:
    """
    Return a dictionary mapping each group name to a collection of results.

    Parameters
    ----------
    results : Results
        Original results
    get_group : Callable(result) -> str
        Function that returns the group for a given result
    
    Returns
    -------
    results_by_group : dict[str, Results]
        Grouped results
    """
    if isinstance(results, dict):
        return {k: group_results(v, get_group) for k, v in results.items()}

    results_by_group = defaultdict(list)
    for result in results:
        results_by_group[get_group(result)].append(result)

    return results_by_group

def load_eval_results(path: str) -> Results:
    """
    Get evaluation results for the given experiment.

    Parameters
    ----------
    path : str
        Path to the experiment directory

    Returns
    -------
    results : Results
        Evaluation results, grouped by dataset and evaluation mode,
        where `results[dataset][eval_mode]` is a list
        of `ray.train.Result` instances
    """
    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(path).resolve().glob('**/eval/tuner.pkl')
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent

    # Load evaluation results
    print('Loading evaluation results from', experiment_path)
    tuner = tune.Tuner.restore(str(experiment_path), trainable=evaluate)
    results = tuner.get_results()
    results = group_results(results, lambda result: get_train_config(result)['dataset'])
    results = group_results(results, lambda result: result.config['eval_mode'])
    return results



### Plotting

def plot_negative_interventions(
    dataset_results: Results, dataset_name: str, groupby: list[str] = ['model_type']):
    """
    Plot negative intervention results.

    Parameters
    ----------
    dataset_results : dict[str, Results]
        Results for the given dataset, grouped by model type and evaluation mode
    dataset_name : str
        Name of the dataset
    groupby : list[str]
        List of train config keys to group by
    """
    for result in dataset_results['neg_intervention']:
        accuracies = np.array(result.metrics['neg_intervention_accs'])
        plt.plot(
            range(len(accuracies)),
            1 - accuracies,
            label=get_group_key(result, groupby)
        )

    plt.xlabel('# of Concepts Intervened')
    plt.ylabel('Classification Error')
    plt.title(f'Negative Interventions: {get_dataset_title(dataset_name)}')
    plt.legend()
    plt.show()

def plot_positive_interventions(
    dataset_results: Results, dataset_name: str, groupby: list[str] = ['model_type']):
    """
    Plot negative intervention results.

    Parameters
    ----------
    dataset_results : dict[str, Results]
        Results for the given dataset, grouped by model type and evaluation mode
    dataset_name : str
        Name of the dataset
    groupby : list[str]
        List of train config keys to group by
    """
    for result in dataset_results['pos_intervention']:
        accuracies = np.array(result.metrics['pos_intervention_accs'])
        plt.plot(
            range(len(accuracies)),
            accuracies,
            label=get_group_key(result, groupby)
        )

    plt.xlabel('# of Concepts Intervened')
    plt.ylabel('Classification Accuracy')
    plt.title(f'Positive Interventions: {get_dataset_title(dataset_name)}')
    plt.legend()
    plt.show()

def plot_random_concepts_residual(
    dataset_results: Results, dataset_name: str, groupby: list[str] = ['model_type']):
    """
    Plot results with randomized concepts and residuals.

    Parameters
    ----------
    dataset_results : dict[str, Results]
        Results for the given dataset, grouped by model type and evaluation mode
    dataset_name : str
        Name of the dataset
    groupby : list[str]
        List of train config keys to group by
    """
    groups = []
    baseline_accuracies = []
    random_concept_accuracies = []
    random_residual_accuracies = []
    for i in range(len(dataset_results['accuracy'])):
        groups.append(get_group_key(dataset_results['accuracy'][i], groupby))
        baseline_accuracies.append(
            dataset_results['accuracy'][i].metrics['test_acc'])
        random_concept_accuracies.append(
            dataset_results['random_concepts'][i].metrics['random_concept_acc'])
        random_residual_accuracies.append(
            dataset_results['random_residual'][i].metrics['random_residual_acc'])

    # Plot
    plt.bar(
        np.arange(len(groups)) - 0.25,
        baseline_accuracies,
        label='Baseline',
        width=0.25,
    )
    plt.bar(
        np.arange(len(groups)),
        random_concept_accuracies,
        label='Random Concepts',
        width=0.25,
    )
    plt.bar(
        np.arange(len(groups)) + 0.25,
        random_residual_accuracies,
        width=0.25,
        label='Random Residual',
    )

    y_min = min([
        *baseline_accuracies,
        *random_concept_accuracies,
        *random_residual_accuracies,
    ])
    plt.xticks(np.arange(len(groups)), groups)
    plt.ylim(max(0, y_min - 0.1), 1)
    plt.ylabel('Classification Accuracy')
    plt.title(f'Random Concepts & Residual: {get_dataset_title(dataset_name)}')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    MODES = ['neg_intervention', 'pos_intervention', 'random']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-dir', type=str, help='Experiment directory')
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
