import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

from data import get_data_loaders
from models import ConceptBottleneckModel, ConceptWhiteningModel
from train import train



### Data

train_loader, test_loader, CONCEPT_DIM = get_data_loaders('cifar100', batch_size=128)
OUTPUT_DIM = 100



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
    ).to('cuda')

def make_whitening_model(residual_dim):
    bottleneck_dim = CONCEPT_DIM + residual_dim
    return ConceptWhiteningModel(
        base_network=make_resnet(bottleneck_dim),
        target_network=make_ffn(bottleneck_dim, OUTPUT_DIM),
        bottleneck_dim=bottleneck_dim,
    ).to('cuda')



if __name__ == '__main__':
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
        num_epochs=100,
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

    # TODO: experiments
    # e.g. concept accuracy, interventions,
    # train model to predict label from residual only, etc ...
