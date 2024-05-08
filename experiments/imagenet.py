import os
import ray

from models import ConceptModel, make_bottleneck_layer
from nn_extensions import Apply
from utils import make_cnn, make_mlp, process_grid_search_tuples
from torch import nn

from models import ConceptMixture


def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config["num_classes"]
    concept_dim = config["concept_dim"]
    residual_dim = config["residual_dim"]
    cnn_type = config.get("cnn_type", "resnet18")
    training_mode = config.get("training_mode", "independent")
    bottleneck_dim = concept_dim + residual_dim

    if config.get("separate_branches", False):
        config["freeze_backbone"] = True
        return ConceptModel(
            concept_network=make_cnn(concept_dim, cnn_type=cnn_type),
            residual_network=make_cnn(residual_dim, cnn_type=cnn_type),
            target_network=nn.Linear(bottleneck_dim, num_classes),
            bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
            **config,
        )

    else:
        # config["freeze_backbone"] = True
        if training_mode == "semi_independent":
            target_network = ConceptMixture(concept_dim, residual_dim, num_classes)
        else:
            target_network = nn.Linear(bottleneck_dim, num_classes)
        return ConceptModel(
            base_network=make_cnn(bottleneck_dim, cnn_type=cnn_type, load_weights=True),
            concept_network=Apply(lambda x: x[..., :concept_dim]),
            residual_network=Apply(lambda x: x[..., concept_dim:]),
            target_network=target_network,
            # make_mlp(
            #     output_dim=num_classes, hidden_dim=1024, num_hidden_layers=2
            # ),  # nn.Linear(bottleneck_dim, num_classes),
            bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
            **config,
        )


def get_config(**kwargs) -> dict:
    config = {
        "model_type": ray.tune.grid_search(
            [
                "latent_residual",
                "decorrelated_residual",
                "iter_norm",
                "mi_residual",
            ]
        ),
        "residual_dim": ray.tune.grid_search([0, 1, 2, 4, 8, 16, 32, 64]),
        "dataset": "imagenet",
        "data_dir": os.environ.get("CONCEPT_DATA_DIR", "./data"),
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        "num_cpus": 4,
        "gpu_memory_per_worker": "11000 MiB",
        "training_mode": "independent",
        "num_epochs": 100,
        "lr": 1e-4,
        "batch_size": 64,
        "alpha": 1.0,
        "beta": 1.0,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 1e-3,
        "cw_alignment_frequency": 20,
        "checkpoint_frequency": 5,
    }
    config.update(kwargs)
    config = process_grid_search_tuples(config)
    return config
