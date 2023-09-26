import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable

from club import CLUB
from datasets.other import MNISTModulo
from utils import (
    train_multiclass_classification,
    cross_correlation,
    get_mi_callback_fn,
)



train_loader = DataLoader(MNISTModulo(train=True), batch_size=64, shuffle=True)
test_loader = DataLoader(MNISTModulo(train=False), batch_size=64, shuffle=False)

INPUT_DIM = 28*28
OUTPUT_DIM = 10
CONCEPT_DIM = 8



def get_base_model(input_dim=28*28, output_dim=10, hidden_dim=128):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )

class ConceptBottleneckModel(nn.Module):

    def __init__(
        self, input_dim: int, output_dim: int, concept_dim: int, residual_dim: int = 0):
        """
        """
        super().__init__()
        self.base_network = nn.Identity()
        self.concept_network = nn.Sequential(
            get_base_model(input_dim, concept_dim), nn.Sigmoid())
        self.residual_network = get_base_model(input_dim, residual_dim)
        self.bottleneck_layer = nn.Identity()
        self.target_network = get_base_model(concept_dim + residual_dim, output_dim)

    def forward(self, x: Tensor, concept_preds: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concept_preds : Tensor or None
            Vector of concept predictions (overrides model's concept predictor)
        """
        if concept_preds is None:
            concept_preds = self.concept_network(self.base_network(x))

        residual = self.residual_network(x)
        bottleneck = self.bottleneck_layer(
            torch.cat([concept_preds.detach(), residual], dim=-1))
        target_preds = self.target_network(bottleneck)
        return concept_preds, residual, target_preds

def joint_training(
    model: ConceptBottleneckModel,
    residual_loss_fn: Callable = lambda r, c: torch.tensor(0),
    alpha: float = 1.0,
    beta: float = 1.0,
    **kwargs):
    """
    Joint training of a concept bottleneck model.
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
        test_loader=test_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]), # (X, concepts)
        loss_fn=loss_fn,
        predict_fn=lambda outputs: outputs[2].argmax(-1), # target_preds
        **kwargs,
    )

    intervention_accuracies = [
        test_negative_interventions(model, num_interventions=i)
        for i in range(1, CONCEPT_DIM + 1)
    ]
    return intervention_accuracies



from tqdm import tqdm
def test_negative_interventions(model, num_interventions):
    def negative_intervention(concept_preds, concepts):
        incorrect_concepts = 1 - concepts   # binary concepts
        intervention_idx = torch.randperm(concept_preds.shape[-1])[:num_interventions]
        concept_preds[:, intervention_idx] = incorrect_concepts[:, intervention_idx]
        return concept_preds

    model.eval()
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for (X, c), y in test_loader:
            #X = model.base_network(X)
            concept_preds = model.concept_network(X)
            concept_preds = negative_intervention(concept_preds, c)
            _, _, target_preds = model(X, concept_preds=concept_preds)
            prediction = target_preds.argmax(-1)
            num_correct += (prediction == y).sum().item()
            num_samples += y.size(0)

        accuracy = num_correct / num_samples
        tqdm.write(f"Intervention Accuracy (n={num_interventions}): {accuracy:.4f}")
    
    return accuracy



if __name__ == '__main__':
    # Without residual
    model = ConceptBottleneckModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        concept_dim=CONCEPT_DIM,
        residual_dim=0,
    )
    y1 = joint_training(model, alpha=1.0)

    # With latent residual
    model = ConceptBottleneckModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        concept_dim=CONCEPT_DIM,
        residual_dim=1,
    )
    y2 = joint_training(model, alpha=1.0)

    # With decorrelated residual
    model = ConceptBottleneckModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        concept_dim=CONCEPT_DIM,
        residual_dim=1,
    )
    y3 = joint_training(
        model,
        residual_loss_fn=lambda r, c: cross_correlation(r, c).square().mean(),
        alpha=1.0,
    )

    # With MI-minimized residual
    model = ConceptBottleneckModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        concept_dim=CONCEPT_DIM,
        residual_dim=1,
    )
    mi_estimator = CLUB(1, CONCEPT_DIM, 128)
    mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=0.001)
    y4 = joint_training(
        model,
        alpha=1.0,
        residual_loss_fn=mi_estimator.forward,
        callback_fn=get_mi_callback_fn(mi_estimator, mi_optimizer),
    )

    # Plot
    import matplotlib.pyplot as plt
    x = list(range(1, CONCEPT_DIM + 1))
    plt.plot(x, y1, label='No residual')
    plt.plot(x, y2, label='Latent residual')
    plt.plot(x, y3, label='Decorrelated residual')
    plt.plot(x, y4, label='MI-minimized residual')
    plt.legend()
    plt.show()
