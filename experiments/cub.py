import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torchvision.models.resnet import resnet18, ResNet18_Weights

from data import get_data_loaders
from evaluation import test_negative_interventions, test_residual_to_label
from models import ConceptBottleneckModel, ConceptWhiteningModel
from train import train



### Data

train_loader, test_loader, CONCEPT_DIM = get_data_loaders('cub', batch_size=64)
OUTPUT_DIM = 200



### Models

def make_ffn(input_dim, output_dim, hidden_dim=256):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )

def make_resnet(output_dim):
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Linear(resnet.fc.in_features, output_dim)
    return resnet

def make_bottleneck_model(residual_dim):
    return ConceptBottleneckModel(
        concept_network=nn.Sequential(make_resnet(CONCEPT_DIM), nn.Sigmoid()),
        residual_network=make_resnet(residual_dim),
        target_network=make_ffn(CONCEPT_DIM + residual_dim, OUTPUT_DIM),
    ).to('cuda:1')

def make_whitening_model(residual_dim):
    bottleneck_dim = CONCEPT_DIM + residual_dim
    return ConceptWhiteningModel(
        base_network=make_resnet(bottleneck_dim),
        target_network=make_ffn(bottleneck_dim, OUTPUT_DIM),
        bottleneck_dim=bottleneck_dim,
    ).to('cuda:1')

def load_models(load_dir: str | Path) -> list[nn.Module]:
    load_dir = Path(load_dir)
    models = []

    models.append(make_bottleneck_model(residual_dim=0))
    models[-1].load_state_dict(
        torch.load(load_dir / 'no_residual.pt'))
    
    models.append(make_bottleneck_model(residual_dim=32))
    models[-1].load_state_dict(
        torch.load(load_dir / 'latent_residual.pt'))
    
    models.append(make_bottleneck_model(residual_dim=32))
    models[-1].load_state_dict(
        torch.load(load_dir / 'decorrelated_residual.pt'))
    
    models.append(make_bottleneck_model(residual_dim=32))
    models[-1].load_state_dict(
        torch.load(load_dir / 'mi_residual.pt'))
    
    models.append(make_whitening_model(residual_dim=32))
    models[-1].load_state_dict(
        torch.load(load_dir / 'whitened_residual.pt'))

    return models



if __name__ == '__main__':
    # TODO: Add argparse
    mode = 'train'
    load_dir = Path('saved_models/CIFAR100/2023-09-28_01_56_41/')

    if mode == 'train':
        models = train(
            make_bottleneck_model_fn=make_bottleneck_model,
            make_whitening_model_fn=make_whitening_model,
            concept_dim=CONCEPT_DIM,
            residual_dim=32,
            train_loader=train_loader,
            test_loader=test_loader,
            save_dir='./saved_models',
            save_interval=10,
            lr=1e-4,
            num_epochs=200,
            bottleneck_alpha=10.0,
            bottleneck_beta=10.0,
            mi_estimator_hidden_dim=256,
            mi_optimizer_lr=0.001,
            whitening_alignment_frequency=20,
        )

        no_residual_model = models[0]
        latent_residual_model = models[1]
        decorrelated_residual_model = models[2]
        mi_residual_model = models[3]
        whitened_residual_model = models[4]

    elif mode == 'intervention':
        results = []
        models = load_models(load_dir)
        for model in models:
            accuracies = test_negative_interventions(
                model, test_loader, CONCEPT_DIM,
                num_interventions=range(0, CONCEPT_DIM + 1),
            )
            results.append(1 - np.array(accuracies))

        # Plot
        x = list(range(0, CONCEPT_DIM + 1))
        plt.plot(x, results[0], label='No residual')
        plt.plot(x, results[1], label='Latent residual')
        plt.plot(x, results[2], label='Decorrelated residual')
        plt.plot(x, results[3], label='MI-minimized residual')
        plt.plot(x, results[4], label='Concept-whitened residual')
        plt.xlabel('# of Concepts Intervened')
        plt.ylabel('Classification Error')
        plt.legend()
        plt.show()

    elif mode == 'residual_to_label':
        results = []
        models = load_models(load_dir)[1:]
        for model in models:
            accuracies = test_residual_to_label(
                model, test_loader, residual_dim=32, num_classes=OUTPUT_DIM)
            results.append(1 - np.array(accuracies))

        # Plot
        residual_types = ['Latent', 'Decorrelated', 'MI-Minimized', 'Concept-Whitened']
        plt.bar(residual_types, results)
        plt.ylabel('Classification Error')
        plt.show()
