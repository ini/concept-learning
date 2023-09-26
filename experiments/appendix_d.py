import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable

from club import CLUB
from data import get_data_loaders
from datasets.pitfalls import DatasetD
from utils import train_multiclass_classification, cross_correlation



# train_loader = DataLoader(CIFAR100(train=True), batch_size=64, shuffle=True)
# test_loader = DataLoader(CIFAR100(train=False), batch_size=64, shuffle=False)

train_loader, test_loader, CONCEPT_DIM = get_data_loaders('cifar100')

for (X, c), y in train_loader:
    print(X.shape, c.shape)
    break

INPUT_DIM = 3*32*32
OUTPUT_DIM = 100





# class ConceptBottleneckModel(nn.Module):

#     def __init__(
#         self, input_dim: int, output_dim: int, concept_dim: int, residual_dim: int = 0):
#         super().__init__()
#         self.concept_network = nn.Sequential(
#             nn.Linear(7, 8), nn.ReLU(),
#             nn.Linear(8, concept_dim), nn.Sigmoid(),
#         )
#         self.residual_network = nn.Sequential(
#             nn.Linear(7, 8), nn.ReLU(),
#             nn.Linear(8, residual_dim),
#         )
#         self.target_network = nn.Sequential(
#             nn.Linear(concept_dim + residual_dim, 4), nn.ReLU(),
#             nn.Linear(4, 2),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         concept_preds = self.concept_network(x)
#         residual = self.residual_network(x)
#         bottleneck = torch.cat([concept_preds, residual], dim=-1)
#         target_preds = self.target_network(bottleneck)
#         return concept_preds, residual, target_preds


import torchvision

def resnet18(output_dim: int, **kwargs):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    return model



class ConceptBottleneckModel(nn.Module):

    def __init__(
        self,
        input_dim: int, output_dim: int,
        concept_dim: int, residual_dim: int = 0,
        hidden_dim: int = 256):
        super().__init__()
        self.base_network = torchvision.models.resnet18(pretrained=True)
        base_dim = self.base_network.fc.in_features
        self.base_network.fc = nn.Identity()

        self.concept_network = nn.Sequential(
            nn.Linear(base_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, concept_dim), nn.Sigmoid(),
        )
        self.residual_network = nn.Sequential(
            nn.Linear(base_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, residual_dim), nn.Sigmoid(),
        )
        self.target_network = nn.Sequential(
            nn.Linear(concept_dim + residual_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            x = self.base_network(x)
        concept_preds = self.concept_network(x)
        residual = self.residual_network(x)
        bottleneck = torch.cat([concept_preds, residual], dim=-1)
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

    test_negative_interventions(model)

def mi_training_step(model, epoch, batch_index, batch):
    (X, concepts), y = batch
    mi_optimizer.zero_grad()
    with torch.no_grad():
        residual = model.residual_network(X)
        concept_preds = model.concept_network(X)

    mi_loss = mi_estimator.learning_loss(residual, concept_preds)
    mi_loss.backward()
    mi_optimizer.step()



from tqdm import tqdm
def test_negative_interventions(model):
    def negative_intervention(concept_preds, concepts, num_interventions):
        incorrect_concepts = 1 - concepts   # binary concepts
        intervention_idx = torch.randperm(concept_preds.shape[-1])[:num_interventions]
        concept_preds[:, intervention_idx] = incorrect_concepts[:, intervention_idx]
        return concept_preds

    model.eval()
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for (X, c), y in test_loader:
            X = model.base_network(X)
            concept_preds = model.concept_network(X)
            concept_preds = negative_intervention(concept_preds, c, num_interventions=2)
            residual = model.residual_network(X)
            bottleneck = torch.cat([concept_preds, residual], dim=-1)
            target_preds = model.target_network(bottleneck)
            prediction = target_preds.argmax(-1)
            num_correct += (prediction == y).sum().item()
            num_samples += y.size(0)

        accuracy = 100 * num_correct / num_samples
        tqdm.write(f"Intervention Accuracy: {accuracy:.2f}%")



### Baseline

# model = nn.Sequential(
#     nn.Linear(7, 8), nn.ReLU(),
#     nn.Linear(8, 6), nn.ReLU(),
#     nn.Linear(6, 4), nn.ReLU(),
#     nn.Linear(4, 2),
# )

# train_multiclass_classification(
#     model,
#     train_loader,
#     test_loader=test_loader,
#     preprocess_fn=lambda batch: (batch[0][0], batch[1]), # (X, y)
#     num_epochs=350,
#     lr=0.001,
# )



# Without residual
model = ConceptBottleneckModel(
    input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, concept_dim=CONCEPT_DIM, residual_dim=0)
joint_training(model, alpha=10)

# With latent residual
model = ConceptBottleneckModel(
    input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, concept_dim=CONCEPT_DIM, residual_dim=1)
joint_training(model, alpha=10)

# With decorrelated residual
model = ConceptBottleneckModel(
    input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, concept_dim=CONCEPT_DIM, residual_dim=1)
joint_training(
    model,
    residual_loss_fn=lambda r, c: cross_correlation(r, c).square().mean(),
    alpha=10,
)

# With MI-minimized residual
model = ConceptBottleneckModel(
    input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, concept_dim=CONCEPT_DIM, residual_dim=1)
mi_estimator = CLUB(1, CONCEPT_DIM, 128)
mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=0.001)
joint_training(
    model,
    alpha=10,
    residual_loss_fn=mi_estimator.forward,
    callback_fn=mi_training_step,
)


# num_concepts = 3
# #num_concepts = 2
# concept_predictor = nn.Sequential(
#     nn.Linear(7, 8), nn.ReLU(),
#     nn.Linear(8, num_concepts), nn.Sigmoid(),
# )
# train_multiclass_classification(
#     concept_predictor,
#     train_loader,
#     loss_fn=lambda data, output, target: nn.BCELoss()(output, target),
#     preprocess_fn=lambda batch: (
#         batch[0][0], batch[0][1][..., :num_concepts]), # (X, concepts)
#     num_epochs=350,
#     lr=0.001,
# )

# concept_predictor.eval()
# target_predictor = nn.Sequential(
#     nn.Linear(num_concepts, 4), nn.ReLU(),
#     nn.Linear(4, 2),
# )
# train_multiclass_classification(
#     target_predictor,
#     train_loader,
#     test_loader=test_loader,
#     preprocess_fn=lambda batch: (
#         concept_predictor(batch[0][0]), batch[1]), # (concept_preds, y)
#     num_epochs=350,
#     lr=0.001,
# )



### Concept Bottleneck Model with Latent Dimension

# num_concepts = 2
# concept_predictor = nn.Sequential(
#     nn.Linear(7, 8), nn.ReLU(),
#     nn.Linear(8, 3), nn.Sigmoid(),
# )
# train_multiclass_classification(
#     concept_predictor,
#     train_loader,
#     loss_fn=lambda data, output, target: nn.BCELoss()(output, target),
#     preprocess_fn=lambda batch: (
#         batch[0][0], batch[0][1][..., :num_concepts]), # (X, concepts)
#     num_epochs=350,
#     lr=0.001,
# )




