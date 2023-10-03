import ray
import torch.nn as nn

from loader import get_data_loaders
from models import ConceptBottleneckModel, ConceptWhiteningModel
from utils import make_mlp



def get_config(**config_override) -> dict:
    config = {
        'dataset': 'pitfalls_random_concepts',
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
    config.update(config_override)

    _, _, _, config['concept_dim'], num_classes = get_data_loaders(
        config['dataset'], data_dir=config['data_dir'], batch_size=config['batch_size'])

    def make_bottleneck_model(residual_dim):
        return ConceptBottleneckModel(
            concept_network=make_mlp(
                config['concept_dim'],
                flatten_input=True,
                output_activation=nn.Sigmoid(),
            ),
            residual_network=make_mlp(residual_dim, flatten_input=True),
            target_network=make_mlp(num_classes),
        )

    def make_whitening_model(residual_dim):
        bottleneck_dim = config['concept_dim'] + residual_dim
        return ConceptWhiteningModel(
            base_network=make_mlp(bottleneck_dim, flatten_input=True),
            target_network=make_mlp(num_classes),
            bottleneck_dim=bottleneck_dim,
        )

    config['make_bottleneck_model_fn'] = make_bottleneck_model
    config['make_whitening_model_fn'] = make_whitening_model

    return config
