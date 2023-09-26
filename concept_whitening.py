import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from data_real import pitfalls_decorrelation_dataset
from lib.iterative_normalization import IterNormRotation as cw_layer
from utils import train_multiclass_classification


def get_base_model(input_dim=28*28, output_dim=10, hidden_dim=128):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ConceptWhiteningModel(nn.Module):

    def __init__(self, concept_dim: int, residual_dim : int = 0):
        super().__init__()
        self.concept_dim = concept_dim
        self.concept_network = get_base_model(output_dim=concept_dim+residual_dim)
        self.cw_layer = cw_layer(concept_dim+residual_dim, activation_mode='mean')
        self.predictor_network = get_base_model(input_dim=concept_dim+residual_dim)

    def forward(self, x):
        x = self.concept_network(x)
        x = self.cw_layer(x[..., None, None])[..., 0, 0]
        x = self.predictor_network(x)
        return x

    def concept_predictions(self, x):
        x = self.concept_network(x)
        x = self.cw_layer(x[..., None, None])
        return x[..., :self.concept_dim, 0, 0]

# Data
train_loader, test_loader, concept_dim = pitfalls_decorrelation_dataset()



print("Creating concept loaders ...")
concept_loaders = [
    torch.utils.data.DataLoader(
        dataset=[
            x for ((x, concepts), y) in train_loader.dataset
            if concepts[concept_index] == 1
        ],
        batch_size=64,
        shuffle=True,
    )
    for concept_index in range(concept_dim)
]
print('Done', '\n')


### Train Concept Whitening Model

def concepts_preprocess_fn(batch):
    (X, concepts), y = batch
    return X, y

def align_concepts(model, epoch, batch_index, batch):
    if (batch_index + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            for concept_index, concept_loader in enumerate(concept_loaders):
                model.cw_layer.mode = concept_index
                for X in concept_loader:
                    X.requires_grad = True
                    model(X)
                    break

                model.cw_layer.update_rotation_matrix(cuda=False)
                model.cw_layer.mode = -1

        model.train()

model = ConceptWhiteningModel(concept_dim=concept_dim)
model = train_multiclass_classification(
    model,
    train_loader,
    test_loader=test_loader,
    num_epochs=20,
    preprocess_fn=concepts_preprocess_fn,
    callback_fn=align_concepts,
)

correct = 0
total = 0
with torch.no_grad():
    for (X, concepts), y in test_loader:
        activations = model.cw_layer(
            model.concept_network(X)[..., None, None])[..., :model.concept_dim, 0, 0]

        predicted_concept_idx = torch.topk(activations, 3).indices.sort().values
        results = torch.stack([
            concepts[i, predicted_concept_idx[i]] for i in range(len(concepts))])
        correct += results.sum()
        total += results.nelement()

accuracy = 100 * correct / total
print(f"Final Concept Accuracy on the test dataset: {accuracy:.2f}%")






### Calculate Intra-Concept Leakage

for concept_index in range(model.concept_dim):
    other_idx = [i for i in range(model.concept_dim) if i != concept_index]
    model_c = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, len(other_idx)))
    
    def transform(batch):
        (X, concepts), y = batch
        with torch.no_grad():
            inputs = model.concept_predictions(X)[..., [concept_index]]
            targets = concepts[..., other_idx].argmax(dim=-1)
            return inputs, targets
        
    model_c = train_multiclass_classification(
        model_c, train_loader, test_loader=test_loader, preprocess_fn=transform, num_epochs=5)

    # running_sum = torch.zeros(len(other_idx))
    # for (X, concepts), y in test_loader:
    #     running_sum += concepts[concepts[concept_index].bool()]
    #     print("RUNNING SUM", running_sum.mean(-1))

    # model_c.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for (X, concepts), y in test_loader:
    #         targets = concepts[..., other_idx].argmax(dim=-1)
    #         outputs = model_c(model.concept_predictions(X)[..., [concept_index]])
    #         predicted = outputs.argmax(dim=-1)
    #         total += targets.size(0)
    #         correct += (predicted == targets).sum().item()

    # print("Accuracy for ", concept_index, correct / total)
