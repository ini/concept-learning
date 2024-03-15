import os
import ray
import torch.nn as nn

from models import ConceptModel, make_bottleneck_layer
from nn_extensions import Apply
from utils import make_cnn, process_grid_search_tuples


def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config["num_classes"]
    concept_dim = config["concept_dim"]
    residual_dim = config["residual_dim"]
    bottleneck_dim = concept_dim + residual_dim
    return ConceptModel(
        base_network=make_cnn(bottleneck_dim, cnn_type="resnet18"),
        concept_network=Apply(lambda x: x[..., :concept_dim]),
        residual_network=Apply(lambda x: x[..., concept_dim:]),
        target_network=nn.Linear(bottleneck_dim, num_classes),
        bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
        **config,
    )


def get_config(**kwargs) -> dict:
    config = {
        # ('model_type', 'beta'): ray.tune.grid_search([
        #     ('latent_residual', 0),
        #     ('decorrelated_residual', 10.0),
        #     ('iter_norm', 0),
        #     ('mi_residual', 1.0),
        # ]),
        "residual_dim": ray.tune.grid_search([0, 1, 2, 4, 8, 16, 32, 64]),
        "dataset": "aa2",
        "data_dir": os.environ.get("CONCEPT_DATA_DIR", "./data"),
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        "training_mode": "independent",
        "num_epochs": 100,
        "lr": 3e-4,
        "batch_size": 64,
        "alpha": 1.0,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 1e-5,
        "cw_alignment_frequency": 20,
        "checkpoint_frequency": 5,
        "gpu_memory_per_worker": "11000 MiB",
        "strategy": ray.train.lightning.RayDDPStrategy(find_unused_parameters=True),
    }
    config.update(kwargs)
    config = process_grid_search_tuples(config)
    return config
