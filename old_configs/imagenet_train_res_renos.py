from ray import tune
from experiments.imagenet import (
    get_config as get_imagenet_config,
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
        "data_dir": "/data/Datasets/imagenet/",
        "ray_storage_dir": "/data/renos/ray_results/",
        "training_mode": "semi_independent",
        "residual_dim": 1024,
        "lr": 3e-4,
        # new params
        # "momentum": 0.9,
        # "weight_decay": 1e-4,
        "lr_scheduler": "cosine annealing",
        "chosen_optim": "adam",
        # "lr_step_size": 5,
        # "lr_gamma": 0.5,
        "num_epochs": 100,
        "freeze_backbone": True,
        "alpha": 1.0,
        "beta": 1.0,
        "mi_estimator_hidden_dim": 1024,
        "mi_optimizer_lr": 0.001,
        "cw_alignment_frequency": 20,
        "num_cpus": 12,
        "num_gpus": 1.0,
        "num_samples": 1,
        "batch_size": 256,
        "checkpoint_frequency": 5,
        "norm_type": None,
        "T_whitening": 3,
    }
    experiment_config.update(kwargs)
    experiment_config = get_imagenet_config(**experiment_config)
    return experiment_config
