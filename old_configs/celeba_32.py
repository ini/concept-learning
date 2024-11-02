from ray import tune
from experiments.celeba import (
    get_config as get_celeba_config,
    make_concept_model,
)


def get_config(**kwargs) -> dict:
    experiment_config = {
        "model_type": "mi_residual",
        "save_dir": "/data/renos/supervised_concept_learning/",
        "data_dir": "/data/Datasets/celeba/",
        "ray_storage_dir": "/data/renos/ray_results/",
        "residual_dim": 32,
        "lr": 0.005,
        "num_epochs": 200,
        "momentum": 0.9,
        "lr_scheduler": "reduce_on_plateau",
        "chosen_optim": "sgd",
        # "lr_scheduler": "cosine annealing",
        # "chosen_optim": "adam",
        "alpha": 1.0,
        "beta": 1.0,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 0.001,
        "cw_alignment_frequency": 20,
        "num_cpus": 8,
        "num_gpus": 1.0,
        "num_samples": 1,
        "batch_size": 64,
        "checkpoint_frequency": 5,
        "norm_type": None,
        "T_whitening": 3,
        "weight_decay": 4e-6,
        "training_mode": "sequential",
        "num_hidden": 0,
        "complete_intervention_weight": tune.grid_search([0.1, 1.0, 2.0, 5.0, 10.0]),
    }
    experiment_config.update(kwargs)
    experiment_config = get_celeba_config(**experiment_config)
    return experiment_config
