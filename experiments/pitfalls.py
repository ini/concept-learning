import os
import ray
import torch.nn as nn

from models import ConceptModel, make_bottleneck_layer
from utils import make_mlp



def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config['num_classes']
    concept_dim = config['concept_dim']
    residual_dim = config['residual_dim']
    bottleneck_dim = concept_dim + residual_dim
    return ConceptModel(
        concept_network=make_mlp(concept_dim, flatten_input=True),
        residual_network=make_mlp(residual_dim, flatten_input=True),
        target_network=nn.Linear(bottleneck_dim, num_classes),
        bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
        **config,
    )

def get_config(**kwargs) -> dict:
    config = {
        'dataset': ray.tune.grid_search([
            'pitfalls_mnist_without_45',
            'pitfalls_random_concepts',
            'pitfalls_synthetic',
            'pitfalls_mnist_123456',
        ]),
        'data_dir': os.environ.get('CONCEPT_DATA_DIR', './data'),
        'save_dir': os.environ.get('CONCEPT_SAVE_DIR', './saved'),
        'model_type': ray.tune.grid_search([
            'no_residual',
            'latent_residual',
            'decorrelated_residual',
            'mi_residual',
            'iter_norm',
            'concept_whitening',
        ]),
        'training_mode': 'independent',
        'residual_dim': 1,
        'num_epochs': 10,
        'lr': 1e-3,
        'batch_size': 64,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-3,
        'whitening_alignment_frequency': 20,
        'checkpoint_frequency': 1,
    }
    config.update(kwargs)
    return config
