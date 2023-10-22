import os
import ray
import torch.nn as nn

from models import ConceptModel, make_bottleneck_layer
from torchvision.models.resnet import resnet18, ResNet18_Weights



### Helper Methods

def make_cnn(output_dim: int):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    return model



### Experiment Module Methods

def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config['num_classes']
    concept_dim = config['concept_dim']
    residual_dim = config['residual_dim']
    bottleneck_dim = concept_dim + residual_dim
    return ConceptModel(
        concept_network=make_cnn(concept_dim),
        residual_network=make_cnn(residual_dim),
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
            'concept_whitening',
        ]),
        'training_mode': 'independent',
        'residual_dim': 8,
        'num_epochs': 100,
        'lr': 1e-4,
        'batch_size': 64,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-3,
        'cw_alignment_frequency': 20,
        'checkpoint_frequency': 5,
    }
    config.update(kwargs)
    return config
