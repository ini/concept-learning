import os
import ray
import torch.nn as nn

from models import ConceptModel, make_bottleneck_layer
from nn_extensions import Apply
from torchvision.models.inception import inception_v3, Inception_V3_Weights



### Helper Methods

def make_cnn(output_dim: int):
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    model.aux_logits = False
    return model



### Experiment Module Methods

def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config['num_classes']
    concept_dim = config['concept_dim']
    residual_dim = config['residual_dim']
    bottleneck_dim = concept_dim + residual_dim
    return ConceptModel(
        base_network=make_cnn(bottleneck_dim),
        concept_network=Apply(lambda x: x[..., :concept_dim]),
        residual_network=Apply(lambda x: x[..., concept_dim:]),
        target_network=nn.Linear(bottleneck_dim, num_classes),
        bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
        **config,
    )

def get_config(**kwargs) -> dict:
    config = {
        'dataset': 'cub',
        'data_dir': os.environ.get('CONCEPT_DATA_DIR', './data'),
        'save_dir': os.environ.get('CONCEPT_SAVE_DIR', './saved'),
        'model_type': ray.tune.grid_search([
            'no_residual',
            'latent_residual',
            'decorrelated_residual',
            'mi_residual',
            'iter_norm',
        ]),
        'training_mode': 'independent',
        'residual_dim': 64,
        'num_epochs': 100,
        'lr': ray.tune.grid_search([k * 1e-4 for k in range(1, 11)]),
        'batch_size': 64,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-4,
        'cw_alignment_frequency': 20,
        'checkpoint_frequency': 5,
        'gpu_memory_per_worker': '11000 MiB',
    }
    config.update(kwargs)
    return config
