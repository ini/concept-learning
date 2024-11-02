from ray import tune
from experiments.cub import (
    get_config as get_cub_config,
    make_concept_model,
)


def get_config(**kwargs) -> dict:
    experiment_config = {
        "model_type": "ccm_eye",  # tune.grid_search(["ccm_eye", "ccm_r"]),
        "save_dir": "/data/renos/supervised_concept_learning/",
        "pretrained_checkpoint": "/data/renos/supervised_concept_learning/baseline_cbm/cub_cbm_checkpoint.pt",
        "data_dir": "/data/Datasets/cub/",
        "ray_storage_dir": "/data/renos/ray_results/",
        "residual_dim": 32,  # tune.grid_search([1, 2, 4, 8, 16, 32]),
        "lr": 0.0003,
        "num_epochs": 300,
        "alpha": 1.0,
        "beta": tune.grid_search([0.001, 0.01, 0.1, 0.5, 1.0]),
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 0.001,
        "cw_alignment_frequency": 20,
        "num_cpus": 8,
        "num_gpus": 1.0,
        "num_samples": 5,
        "batch_size": 64,
        "checkpoint_frequency": 5,
        "norm_type": None,
        "T_whitening": 3,
    }
    experiment_config.update(kwargs)
    experiment_config = get_cub_config(**experiment_config)
    return experiment_config
