import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable

from club import CLUB
from datasets.other import MNISTModulo
from models import ConceptBottleneckModel, ConceptWhiteningModel
from utils import (
    accuracy,
    train_multiclass_classification,
    cross_correlation,
    get_cw_callback_fn,
    get_mi_callback_fn,
    concepts_preprocess_fn,
)



### Data

train_loader = DataLoader(MNISTModulo(train=True), batch_size=64, shuffle=True)
test_loader = DataLoader(MNISTModulo(train=False), batch_size=64, shuffle=False)

INPUT_DIM = 28*28
OUTPUT_DIM = 10
CONCEPT_DIM = 5



### Models

def get_base_model(input_dim, output_dim, hidden_dim=128):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )

def bottleneck_model(
    input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, concept_dim=CONCEPT_DIM, residual_dim=0):
    return ConceptBottleneckModel(
        concept_network=nn.Sequential(
            get_base_model(input_dim, concept_dim), nn.Sigmoid()),
        residual_network=get_base_model(input_dim, residual_dim),
        target_network=get_base_model(concept_dim + residual_dim, output_dim),
    )

def whitening_model(
    input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, concept_dim=CONCEPT_DIM, residual_dim=0):
    bottleneck_dim = concept_dim + residual_dim
    return ConceptWhiteningModel(
        base_network=get_base_model(input_dim, bottleneck_dim),
        target_network=get_base_model(bottleneck_dim, output_dim),
        bottleneck_dim=bottleneck_dim,
    )



### Training

def train_bottleneck_joint(
    model: ConceptBottleneckModel,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    residual_loss_fn: Callable = lambda r, c: torch.tensor(0),
    alpha: float = 10.0,
    beta: float = 1.0,
    **kwargs):
    """
    Joint training of a concept bottleneck model.

    Parameters
    ----------
    model : ConceptBottleneckModel
        Model to train
    residual_loss_fn : Callable(residual, concept_preds) -> Tensor
        Function to compute the residual loss
    alpha : float
        Weight of the concept loss
    beta : float
        Weight of the residual loss
    **kwargs
        Additional arguments to pass to `train_multiclass_classification()`
    """
    def loss_fn(data, output, target):
        _, concepts = data
        concept_preds, residual, target_preds = output
        concept_target = concepts[..., :concept_preds.shape[-1]]
        concept_loss = nn.BCELoss()(concept_preds, concept_target)
        residual_loss = residual_loss_fn(residual, concept_preds)
        target_loss = nn.CrossEntropyLoss()(target_preds, target)
        return (alpha * concept_loss) + (beta * residual_loss) + target_loss

    train_multiclass_classification(
        model,
        train_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]),
        loss_fn=loss_fn,
        **kwargs,
    )

    if test_loader is not None:
        print(
            'Test Classification Accuracy:',
            accuracy(
                model, test_loader,
                preprocess_fn=lambda batch: (batch[0][0], batch[1]),
                predict_fn=lambda outputs: outputs[2].argmax(-1),
            )
        )
        print(
            'Test Concept Accuracy:',
            accuracy(
                model, test_loader,
                preprocess_fn=lambda batch: (batch[0][0], batch[0][1]),
                predict_fn=lambda outputs: (outputs[0] > 0.5).float(),
            )
        )

        try:
            print("Training to predict target from residual only")
            residual_to_label_model = get_base_model(RESIDUAL_DIM, OUTPUT_DIM)
            train_multiclass_classification(
                residual_to_label_model, train_loader,
                preprocess_fn=lambda batch: (model(batch[0][0])[1], batch[1]),
            )
            print(
                'Test (Residual -> Label) Accuracy:',
                accuracy(
                    residual_to_label_model, test_loader,
                    preprocess_fn=lambda batch: (model(batch[0][0])[1], batch[1]),
                )
            )
        except Exception as e:
            print(e)

def train_whitening(
    model: ConceptWhiteningModel,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    **kwargs):
    train_multiclass_classification(
        model, train_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]),
        callback_fn=get_cw_callback_fn(
            train_loader, CONCEPT_DIM, alignment_frequency=20),
        **kwargs,
    )

    if test_loader is not None:
        print(
            'Test Classification Accuracy:',
            accuracy(
                model, test_loader,
                preprocess_fn=lambda batch: (batch[0][0], batch[1]),
            )
        )

        residual_to_label_model = get_base_model(RESIDUAL_DIM, OUTPUT_DIM)
        print("Training to predict target from residual only")

        def preprocess_fn(batch):
            with torch.no_grad():
                (X, c), y = batch
                residual = model.activations(X)[:, c.shape[1]:]
                return residual, y

        train_multiclass_classification(
            residual_to_label_model, train_loader, preprocess_fn=preprocess_fn)
        print(
            'Test (Residual -> Label) Accuracy:',
            accuracy(residual_to_label_model, test_loader, preprocess_fn=preprocess_fn)
        )



### Interventions

def test_negative_interventions(model, num_interventions):
    def negative_intervention(concept_preds, concepts):
        if isinstance(model, ConceptBottleneckModel):
            incorrect_concepts = 1 - concepts   # binary concepts
        elif isinstance(model, ConceptWhiteningModel):
            incorrect_concepts = 1 - 2 * concepts # concept activations

        intervention_idx = torch.randperm(concept_preds.shape[-1])[:num_interventions]
        concept_preds[:, intervention_idx] = incorrect_concepts[:, intervention_idx]
        return concept_preds

    model.eval()
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for (X, c), y in test_loader:
            if isinstance(model, ConceptBottleneckModel):
                X = model.base_network(X)
                concept_preds = model.concept_network(X)
                concept_preds = negative_intervention(concept_preds, c)
                _, _, target_preds = model(X, concept_preds=concept_preds)

            elif isinstance(model, ConceptWhiteningModel):
                X = model.base_network(X)
                bottleneck = X
                while bottleneck.ndim < 4:
                    bottleneck = bottleneck.unsqueeze(-1)
                bottleneck = model.bottleneck_layer(bottleneck).view(X.shape)
                bottleneck[:, :CONCEPT_DIM] = negative_intervention(
                    bottleneck[:, :CONCEPT_DIM], c)
                target_preds = model.target_network(bottleneck)

            prediction = target_preds.argmax(-1)
            num_correct += (prediction == y).sum().item()
            num_samples += y.size(0)

        accuracy = num_correct / num_samples
        tqdm.write(f"Intervention Accuracy (n={num_interventions}): {accuracy:.4f}")

    return accuracy

def test_negative_interventions_multiple(model, values):
    return [
        test_negative_interventions(model, num_interventions=i)
        for i in values
    ]



if __name__ == '__main__':
    dataset_name = train_loader.dataset.__class__.__name__
    RESIDUAL_DIM = 1

    # Without residual
    model = bottleneck_model(residual_dim=0)
    train_bottleneck_joint(
        model, train_loader,
        test_loader=test_loader,
        save_path=f'./saved_models/{dataset_name}/no_residual.pt',
    )
    y1 = test_negative_interventions_multiple(model, values=range(0, CONCEPT_DIM + 1))

    # With latent residual
    model = bottleneck_model(residual_dim=RESIDUAL_DIM)
    train_bottleneck_joint(
        model, train_loader,
        test_loader=test_loader,
        save_path=f'./saved_models/{dataset_name}/latent_residual.pt',
    )
    y2 = test_negative_interventions_multiple(model, values=range(0, CONCEPT_DIM + 1))

    # With decorrelated residual
    model = bottleneck_model(residual_dim=RESIDUAL_DIM)
    train_bottleneck_joint(
        model, train_loader,
        test_loader=test_loader,
        residual_loss_fn=lambda r, c: cross_correlation(r, c).square().mean(),
        save_path=f'./saved_models/{dataset_name}/decorrelated_residual.pt',
    )
    y3 = test_negative_interventions_multiple(model, values=range(0, CONCEPT_DIM + 1))

    # With MI-minimized residual
    model = bottleneck_model(residual_dim=RESIDUAL_DIM)
    mi_estimator = CLUB(RESIDUAL_DIM, CONCEPT_DIM, 128)
    mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=0.001)
    train_bottleneck_joint(
        model, train_loader,
        test_loader=test_loader,
        residual_loss_fn=mi_estimator.forward,
        callback_fn=get_mi_callback_fn(mi_estimator, mi_optimizer),
        save_path=f'./saved_models/{dataset_name}/mi_residual.pt',
    )
    y4 = test_negative_interventions_multiple(model, values=range(0, CONCEPT_DIM + 1))

    # With concept-whitened residual
    model = whitening_model(residual_dim=1)
    train_whitening(
        model, train_loader,
        test_loader=test_loader,
        save_path=f'./saved_models/{dataset_name}/whitened_residual.pt',
    )
    y5 = test_negative_interventions_multiple(model, values=range(0, CONCEPT_DIM + 1))

    # Plot
    import matplotlib.pyplot as plt
    x = list(range(0, CONCEPT_DIM + 1))
    plt.plot(x, y1, label='No residual')
    plt.plot(x, y2, label='Latent residual')
    plt.plot(x, y3, label='Decorrelated residual')
    plt.plot(x, y4, label='MI-minimized residual')
    plt.plot(x, y5, label='Concept-whitened residual')
    plt.legend()
    plt.show()
