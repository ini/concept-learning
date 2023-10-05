import argparse
import matplotlib.pyplot as plt
import numpy as np
import ray

from collections import defaultdict
from pathlib import Path
from ray import tune
from typing import Callable, Iterable

from evaluate import evaluate
from utils import disable_ray_storage_context

Results = Iterable[ray.train.Result] | dict[str, 'Results'] # typing



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

def get_eval_results(experiment_path: str) -> Results:
    """
    Get evaluation results for the given experiment.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory

    Returns
    -------
    results : Results
        Evaluation results, grouped by dataset, model type, and evaluation mode,
        where `results[dataset][model_type][eval_mode]` is a list
        of `ray.train.Result` instances
    """
    disable_ray_storage_context()

    experiment_path = Path(experiment_path).resolve()
    tuner = tune.Tuner.restore(str(experiment_path / 'eval'), trainable=evaluate)
    results = tuner.get_results()
    results = group_results(
        results, lambda result: result.config['train_result'].config['dataset'])
    results = group_results(
        results, lambda result: result.config['train_result'].config['model_type'])
    results = group_results(
        results, lambda result: result.config['eval_mode'])

    return results



### Plotting

def plot_negative_interventions(dataset_results: Results, dataset_name: str):
    """
    Plot negative intervention results.

    Parameters
    ----------
    dataset_results : dict[str, Results]
        Results for the given dataset, grouped by model type and evaluation mode
    dataset_name : str
        Name of the dataset
    """
    for model_type, model_results in dataset_results.items():
        intervention_results = model_results['neg_intervention']
        intervention_results = group_results(
            intervention_results, lambda result: result.config['num_interventions'])
        num_interventions = sorted(intervention_results.keys())
        accuracies = np.array([
            intervention_results[n][0].metrics['neg_intervention_acc']
            for n in num_interventions
        ])
        plt.plot(num_interventions, 1 - accuracies, label=model_type)

    plt.xlabel('# of Concepts Intervened')
    plt.ylabel('Classification Error')
    plt.title(f'Negative Interventions: {get_dataset_title(dataset_name)}')
    plt.legend()
    plt.show()

def plot_positive_interventions(dataset_results: Results, dataset_name: str):
    """
    Plot positive intervention results.

    Parameters
    ----------
    dataset_results : dict[str, Results]
        Results for the given dataset, grouped by model type and evaluation mode
    dataset_name : str
        Name of the dataset
    """
    for model_type, model_results in dataset_results.items():
        intervention_results = model_results['pos_intervention']
        intervention_results = group_results(
            intervention_results, lambda result: result.config['num_interventions'])
        num_interventions = sorted(intervention_results.keys())
        accuracies = np.array([
            intervention_results[n][0].metrics['pos_intervention_acc']
            for n in num_interventions
        ])
        plt.plot(num_interventions, accuracies, label=model_type)

    plt.xlabel('# of Concepts Intervened')
    plt.ylabel('Classification Accuracy')
    plt.title(f'Negative Interventions: {get_dataset_title(dataset_name)}')
    plt.legend()
    plt.show()

def plot_random_concepts_residual(dataset_results: Results, dataset_name: str):
    """
    Plot results with randomized concepts and residuals.

    Parameters
    ----------
    dataset_results : dict[str, Results]
        Results for the given dataset, grouped by model type and evaluation mode
    dataset_name : str
        Name of the dataset
    """
    baseline_accuracies = []
    random_concept_accuracies = []
    random_residual_accuracies = []
    for model_type in sorted(dataset_results.keys()):
        model_results = dataset_results[model_type]
        baseline_accuracies.append(
            model_results['accuracy'][0].metrics['test_acc'])
        random_concept_accuracies.append(
            model_results['random_concepts'][0].metrics['random_concept_acc'])
        random_residual_accuracies.append(
            model_results['random_residual'][0].metrics['random_residual_acc'])

    # Plot
    plt.bar(
        np.arange(len(dataset_results)) - 0.25,
        baseline_accuracies,
        label='Baseline',
        width=0.25,
    )
    plt.bar(
        np.arange(len(dataset_results)),
        random_concept_accuracies,
        label='Random Concepts',
        width=0.25,
    )
    plt.bar(
        np.arange(len(dataset_results)) + 0.25,
        random_residual_accuracies,
        width=0.25,
        label='Random Residual',
    )

    y_min = min([
        *baseline_accuracies,
        *random_concept_accuracies,
        *random_residual_accuracies,
    ])
    plt.xticks(np.arange(len(dataset_results)), sorted(dataset_results.keys()))
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

    args = parser.parse_args()
    eval_results = get_eval_results(args.exp_dir)

    # Plot results for each dataset
    for dataset_name, dataset_results in eval_results.items():
        for mode in args.mode:
            if mode == 'neg_intervention':
                plot_negative_interventions(dataset_results, dataset_name)
            elif mode == 'pos_intervention':
                plot_positive_interventions(dataset_results, dataset_name)
            elif mode == 'random':
                plot_random_concepts_residual(dataset_results, dataset_name)
