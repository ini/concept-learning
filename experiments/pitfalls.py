import ray
import torch.nn as nn

from models import ConceptBottleneckModel, ConceptWhiteningModel
from utils import make_mlp



def make_bottleneck_model(config):
    return ConceptBottleneckModel(
        concept_network=make_mlp(
            config['concept_dim'],
            flatten_input=True,
            output_activation=nn.Sigmoid(),
        ),
        residual_network=make_mlp(config['residual_dim'], flatten_input=True),
        target_network=make_mlp(config['num_classes']),
        config=config,
    )

def make_whitening_model(config):
    bottleneck_dim = config['concept_dim'] + config['residual_dim']
    return ConceptWhiteningModel(
        base_network=make_mlp(bottleneck_dim, flatten_input=True),
        target_network=make_mlp(config['num_classes']),
        bottleneck_dim=bottleneck_dim,
    )

def get_config(**kwargs) -> dict:
    config = {
        'dataset': ray.tune.grid_search([
            'pitfalls_mnist_without_45',
            'pitfalls_random_concepts',
            'pitfalls_synthetic',
            'pitfalls_mnist_123456',
        ]),
        'data_dir': './data',
        'save_dir': './saved',
        'model_type': ray.tune.grid_search([
            'no_residual',
            'latent_residual',
            'decorrelated_residual',
            'mi_residual',
            'whitened_residual',
        ]),
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
