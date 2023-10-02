import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from torchvision.models.resnet import resnet18, ResNet18_Weights

from evaluation import (
    test_negative_interventions,
    test_random_concepts,
    test_random_residual,
)
from loader import get_data_loaders
from models import ConceptBottleneckModel, ConceptWhiteningModel
from train import train, load_models
from utils import make_ffn, concept_model_accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'intervention', 'random'],
        help='Mode to run',
    )
    parser.add_argument(
        '--device', type=str, default='cpu', help='Device to use')
    parser.add_argument(
        '--save-dir', type=str, help='Directory to save models to')
    parser.add_argument(
        '--load-dir', type=str, help='Directory to load saved models from')
    parser.add_argument(
        '--num-epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--alpha', type=float, default=1.0, help='Weight for concept loss')
    parser.add_argument(
        '--beta', type=float, default=1.0, help='Weight for residual loss')
    args = parser.parse_args()



    ### Data

    train_loader, test_loader, CONCEPT_DIM, NUM_CLASSES = get_data_loaders(
        'cifar100', batch_size=64)
    


    ### Models

    def make_resnet(output_dim):
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, output_dim)
        return resnet

    def make_bottleneck_model(residual_dim):
        return ConceptBottleneckModel(
            concept_network=nn.Sequential(make_resnet(CONCEPT_DIM), nn.Sigmoid()),
            residual_network=make_resnet(residual_dim),
            target_network=make_ffn(NUM_CLASSES),
        ).to(args.device)

    def make_whitening_model(residual_dim):
        bottleneck_dim = CONCEPT_DIM + residual_dim
        return ConceptWhiteningModel(
            base_network=make_resnet(bottleneck_dim),
            target_network=make_ffn(NUM_CLASSES),
            bottleneck_dim=bottleneck_dim,
        ).to(args.device)

    ### Experiments

    RESIDUAL_DIM = 32 # if applicable

    if args.mode == 'train':
        models = train(
            make_bottleneck_model_fn=make_bottleneck_model,
            make_whitening_model_fn=make_whitening_model,
            concept_dim=CONCEPT_DIM,
            residual_dim=RESIDUAL_DIM,
            train_loader=train_loader,
            test_loader=test_loader,
            save_dir='./saved_models',
            save_interval=10,
            lr=args.lr,
            num_epochs=args.num_epochs,
            bottleneck_alpha=args.alpha,
            bottleneck_beta=args.beta,
            mi_estimator_hidden_dim=256,
            mi_optimizer_lr=0.001,
            whitening_alignment_frequency=20,
        )

    if args.mode == 'intervention':
        results = {}
        models = load_models(
            args.load_dir, make_bottleneck_model, make_whitening_model, RESIDUAL_DIM)
        for model_name, model in models.items():
            accuracies = test_negative_interventions(
                model, test_loader, CONCEPT_DIM,
                num_interventions=range(0, CONCEPT_DIM + 1),
            )
            results[model_name] = 1 - np.array(accuracies)

        # Plot
        x = list(range(0, CONCEPT_DIM + 1))
        for model_name, results in results.items():
            plt.plot(x, results, label=model_name)
        plt.xlabel('# of Concepts Intervened')
        plt.ylabel('Classification Error')
        plt.legend()
        plt.show()

    elif args.mode == 'random':
        baseline_results, random_concepts_results, random_residual_results = [], [], []
        models = load_models(
            args.load_dir, make_bottleneck_model, make_whitening_model, RESIDUAL_DIM)
        for model_name in sorted(models.keys()):
            print('\n', 'Model:', model_name)
            baseline_acc = concept_model_accuracy(models[model_name], test_loader)
            random_concepts_acc = test_random_concepts(
                models[model_name], test_loader, RESIDUAL_DIM)
            random_residual_acc = test_random_residual(
                models[model_name], test_loader, RESIDUAL_DIM)
            baseline_results.append(1 - baseline_acc)
            random_concepts_results.append(1 - random_concepts_acc)
            random_residual_results.append(1 - random_residual_acc)

        # Plot
        plt.bar(
            np.arange(len(models)) - 0.25,
            baseline_results,
            label='Baseline',
            width=0.25,
        )
        plt.bar(
            np.arange(len(models)),
            random_concepts_results,
            label='Random Concepts',
            width=0.25,
        )
        plt.bar(
            np.arange(len(models)) + 0.25,
            random_residual_results,
            width=0.25,
            label='Random Residual',
        )
        plt.xticks(np.arange(len(models)), sorted(models.keys()))
        plt.ylabel('Classification Error')
        plt.legend()
        plt.show()
