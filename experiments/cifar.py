import ray
import torch.nn as nn

from loader import get_data_loaders
from models import ConceptBottleneckModel, ConceptWhiteningModel
from torchvision.models.resnet import resnet18, ResNet18_Weights
from utils import make_mlp



def make_resnet(output_dim):
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Linear(resnet.fc.in_features, output_dim)
    return resnet

def get_config(**kwargs) -> dict:
    config = {
        'dataset': 'cifar100',
        'model_type': ray.tune.grid_search([
            'no_residual',
            'latent_residual',
            'decorrelated_residual',
            'mi_residual',
            'whitened_residual',
        ]),
        'residual_dim': 32,
        'num_epochs': 100,
        'lr': 1e-4,
        'batch_size': 64,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-3,
        'whitening_alignment_frequency': 20,
        'checkpoint_frequency': 5,
    }
    config.update(kwargs)

    _, _, _, config['concept_dim'], num_classes = get_data_loaders(
        config['dataset'], config['data_dir'])

    def make_bottleneck_model(residual_dim):
        return ConceptBottleneckModel(
            concept_network=nn.Sequential(
                make_resnet(config['concept_dim']), nn.Sigmoid()),
            residual_network=make_resnet(residual_dim),
            target_network=make_mlp(num_classes),
        )

    def make_whitening_model(residual_dim):
        bottleneck_dim = config['concept_dim'] + residual_dim
        return ConceptWhiteningModel(
            base_network=make_resnet(bottleneck_dim),
            target_network=make_mlp(num_classes),
            bottleneck_dim=bottleneck_dim,
        )

    config['make_bottleneck_model_fn'] = make_bottleneck_model
    config['make_whitening_model_fn'] = make_whitening_model

    return config
