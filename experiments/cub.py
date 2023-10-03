import ray
import torch.nn as nn

from loader import get_data_loaders
from models import ConceptBottleneckModel, ConceptWhiteningModel
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from utils import make_mlp



class InceptionV3(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        return self.model.forward(x).logits        


def make_bottleneck_model(config):
    return ConceptBottleneckModel(
        concept_network=nn.Sequential(
            InceptionV3(config['concept_dim']), nn.Sigmoid()),
        residual_network=InceptionV3(config['residual_dim']),
        target_network=make_mlp(config['num_classes']),
        config=config,
    )

def make_whitening_model(config):
    bottleneck_dim = config['concept_dim'] + config['residual_dim']
    return ConceptWhiteningModel(
        base_network=InceptionV3(bottleneck_dim),
        target_network=make_mlp(config['num_classes']),
        bottleneck_dim=bottleneck_dim,
    )

def get_config(**kwargs) -> dict:
    config = {
        'dataset': 'cub',
        'make_bottleneck_model_fn': make_bottleneck_model,
        'make_whitening_model_fn': make_whitening_model,
        'model_type': ray.tune.grid_search([
            'no_residual',
            'latent_residual',
            'decorrelated_residual',
            'mi_residual',
            'whitened_residual',
        ]),
        'residual_dim': 32,
        'num_epochs': 100,
        'lr': 1e-3,
        'batch_size': 64,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-3,
        'whitening_alignment_frequency': 20,
        'checkpoint_frequency': 5,
    }
    config.update(kwargs)

    _, _, _, config['concept_dim'], config['num_classes'] = get_data_loaders(
        config['dataset'], config['data_dir'])

    config['make_bottleneck_model_fn'] = make_bottleneck_model
    config['make_whitening_model_fn'] = make_whitening_model

    return config
