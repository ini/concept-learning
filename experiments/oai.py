import os
import ray
import torch.nn as nn

from models import ConceptModel, make_bottleneck_layer
from nn_extensions import Apply
from torchvision.models.resnet import resnet18, ResNet18_Weights
from utils import make_mlp



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
        base_network=make_cnn(bottleneck_dim),
        concept_network=Apply(lambda x: x[..., :concept_dim]),
        residual_network=Apply(lambda x: x[..., concept_dim:]),
        target_network=make_mlp(num_classes, num_hidden_layers=2, hidden_dim=50),
        bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
        **config,
    )

def get_config(**kwargs) -> dict:
    config = {
        'dataset': 'oai',
        'data_dir': os.environ.get('CONCEPT_DATA_DIR', './data'),
        'save_dir': os.environ.get('CONCEPT_SAVE_DIR', './saved'),
        'model_type': ray.tune.grid_search([
            'no_residual',
            'latent_residual',
            'decorrelated_residual',
            'mi_residual',
            'iter_norm',
        ]),
        'num_gpus': 1.0,
        'training_mode': 'independent',
        'residual_dim': ray.tune.grid_search([0, 1, 2, 4, 8, 16, 32, 64]),
        'num_epochs': 100,
        'lr': 1e-4,
        'batch_size': 8,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-5,
        'cw_alignment_frequency': 20,
        'checkpoint_frequency': 5,
        'gpu_memory_per_worker': '9000 MiB',
    }
    config.update(kwargs)
    return config
