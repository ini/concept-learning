from ray import tune
from experiments.cifar import (
    get_config as get_cifar_config,
    make_bottleneck_model,
    make_whitening_model,
)



def get_config(**kwargs) -> dict:
    experiment_config = {
        "model_type" : "concept_whitening",
        "save_dir" : "/data/renos/supervised_concept_learning/",
        "data_dir" : "/data/Datasets/cifar/",
        "ray_storage_dir" : "/data/renos/ray_results/",
        "residual_dim" : 2,
        "lr" : 1e-4,
        "num_epochs" : 1000,
        "alpha" : 1.0,
        "beta" : 1.0,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr" : 0.001,
        "whitening_alignment_frequency": 20,
        "num_gpus" : 0.2,
        "num_samples" : 1,
        "batch_size" : 64,
        "checkpoint_freq" : 5,
    }
    experiment_config.update(kwargs)
    experiment_config = get_cifar_config(**experiment_config)
    return experiment_config
