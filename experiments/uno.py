import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

from datasets.pitfalls import MNIST_45
from utils import train_multiclass_classification



def get_base_model(input_dim: int, output_dim: int, hidden_dim: int = 128):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )

class ConceptBottleneckModel(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, concept_dim: int):
        super().__init__()
        self.concept_predictor = nn.Sequential(
            get_base_model(input_dim, concept_dim),
            nn.Sigmoid(),
        )
        self.target_predictor = get_base_model(concept_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        concept_preds = self.concept_predictor(x)
        target_preds = self.target_predictor(concept_preds)
        return concept_preds, target_preds




concept_dim = 2
train_loader = DataLoader(MNIST_45(train=True), batch_size=64, shuffle=True)
test_loader = DataLoader(MNIST_45(train=False), batch_size=64, shuffle=False)



def sequential_training():
    model = ConceptBottleneckModel(input_dim=28*28, output_dim=2, concept_dim=2)

    # Train concept predictor
    train_multiclass_classification(
        model.concept_predictor,
        train_loader,
        loss_fn=lambda data, output, target: nn.BCELoss()(output, target),
        preprocess_fn=lambda batch: (batch[0][0], batch[0][1]), # (X, concepts)
        num_epochs=2,
        lr=0.001,
    )

    # Train target predictor
    model.concept_predictor.eval()
    train_multiclass_classification(
        model.target_predictor,
        train_loader,
        test_loader=test_loader,
        preprocess_fn=lambda batch: (
            model.concept_predictor(batch[0][0]), batch[1]), # (concept_preds, y)
        num_epochs=2,
        lr=0.001,
    )

def joint_training():
    model = ConceptBottleneckModel(input_dim=28*28, output_dim=2, concept_dim=2)
    
    def loss_fn(data, output, target):
        concept_preds, target_preds = output
        concept_loss = nn.BCELoss()(concept_preds, data[1])
        target_loss = nn.CrossEntropyLoss()(target_preds, target)
        return 1e-3 * concept_loss + target_loss

    train_multiclass_classification(
        model,
        train_loader,
        test_loader=test_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]), # (X, concepts)
        loss_fn=loss_fn,
        predict_fn=lambda output: output[1].argmax(-1), # target_preds
        num_epochs=10,
        lr=0.001,
    )

#sequential_training()
joint_training()
