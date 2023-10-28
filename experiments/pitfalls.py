import copy
import os
import ray
import torch.nn as nn

from nn_extensions import Dummy
from models import ConceptModel, make_bottleneck_layer
from utils import make_mlp



def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config['num_classes']
    concept_dim = config['concept_dim']
    residual_dim = config['residual_dim']
    bottleneck_dim = concept_dim + residual_dim

    # Create concept and target networks
    if config['dataset'] == 'pitfalls_synthetic':
        concept_network = nn.Sequential(
            nn.LazyLinear(8), nn.ReLU(),
            nn.LazyLinear(6), nn.ReLU(),
            nn.LazyLinear(concept_dim),
        )
        target_network = nn.Sequential(
            nn.LazyLinear(4), nn.ReLU(), nn.LazyLinear(num_classes))

    elif config['dataset'] == 'pitfalls_mnist_123456':
        config.setdefault('norm_type', 'batch_norm') # bottleneck layer default
        conv_block = lambda n: nn.Sequential(
            nn.LazyConv2d(n, kernel_size=3), nn.BatchNorm2d(n), nn.ReLU())
        concept_network = nn.Sequential(
            conv_block(8),
            conv_block(16), nn.MaxPool2d(2, stride=2),
            conv_block(32),
            conv_block(16), nn.MaxPool2d(2, stride=2),
            nn.LazyConv2d(8, kernel_size=3),
            nn.Flatten(), nn.AdaptiveAvgPool1d(concept_dim),
        )
        target_network = nn.LazyLinear(num_classes)

    else:
        concept_network = make_mlp(
            concept_dim, hidden_dim=128, num_hidden_layers=2, flatten_input=True)
        target_network = make_mlp(
            num_classes, hidden_dim=128, num_hidden_layers=2)

    # Create residual network with same architecture as concept network
    if residual_dim > 0:
        residual_network = copy.deepcopy(concept_network)
        residual_network[-1] = residual_network[-1].__class__(residual_dim)
    else:
        residual_network = Dummy()

    return ConceptModel(
        concept_network=concept_network,
        residual_network=residual_network,
        target_network=target_network,
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
        'training_mode': 'sequential',
        'residual_dim': 1,
        'num_epochs': 10,
        'lr': ray.tune.grid_search([0.1, 0.01, 0.001]),
        'batch_size': 64,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-3,
        'cw_alignment_frequency': 20,
        'checkpoint_frequency': 1,
    }
    config.update(kwargs)
    return config
