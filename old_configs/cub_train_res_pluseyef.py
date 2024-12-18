from ray import tune
from experiments.cub import (
    get_config as get_cub_config,
    make_concept_model,
)


def get_config(**kwargs) -> dict:
    experiment_config = {
        "model_type": tune.grid_search(
            [
                "latent_residual",
                "decorrelated_residual",
                "mi_residual",
            ]
        ),
        "save_dir": "/data/renos/supervised_concept_learning/",
        "data_dir": "/data/Datasets/cub/",
        "ray_storage_dir": "/data/renos/ray_results/",
        "residual_dim": tune.grid_search([1, 2, 4, 8, 16, 32, 64]),
        "lr": 0.0003,
        "num_epochs": 300,
        "alpha": 1.0,
        "beta": 1.0,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 0.001,
        "cw_alignment_frequency": 20,
        "num_cpus": 4,
        "num_gpus": 0.15,
        "num_samples": 5,
        "batch_size": 64,
        "checkpoint_frequency": 5,
        "norm_type": None,
        "reg_type": "eye",
        "reg_gamma": 5e-4,
        "T_whitening": 3,
        "weight_decay": 4e-6,
    }
    experiment_config.update(kwargs)
    experiment_config = get_cub_config(**experiment_config)
    return experiment_config
