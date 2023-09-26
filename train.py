import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from club import CLUB


class FilteredMNISTDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        digits=set(range(10)),
    ):
        self.original_dataset = datasets.MNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.filtered_indices = self.filter_indices(digits)
        #self.filtered_indices = list(range(len(self.original_dataset)))

    def filter_indices(self, digits):
        filtered_indices = []
        for idx in range(len(self.original_dataset)):
            x, y = self.original_dataset[idx]
            if y in digits:
                filtered_indices.append(idx)

        return filtered_indices

    def __getitem__(self, index):
        index = self.filtered_indices[index]
        x, y = self.original_dataset[index]
        concepts = torch.tensor([y == 4, y == 5]).float()
        return (x, concepts), y % 2

    def __len__(self):
        return len(self.filtered_indices)




# Define a transform to preprocess the data (you can customize this)
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
train_dataset = FilteredMNISTDataset(
    root='./data',
    train=True,
    transform=transform,
    download=True,
    digits={4, 5},
)

train2_dataset = FilteredMNISTDataset(
    root='./data',
    train=True,
    transform=transform,
    download=True,
    digits={4, 5},
)

test_dataset = FilteredMNISTDataset(
    root='./data',
    train=False,
    transform=transform,
    download=True,
    digits = set(range(10)) - {4, 5},
)

# Create DataLoader instances for batching
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple neural network model for multiclass classification
class ConceptModel(nn.Module):

    def __init__(
        self,
        input_dim=28*28, output_dim=1, hidden_dim=128,
        concept_dim=2, residual_dim=8,
    ):
        super().__init__()

        self.concept_network = None
        if concept_dim > 0:
            self.concept_network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, concept_dim), nn.Sigmoid(),
            )

        self.residual_network = None
        if residual_dim > 0:
            self.residual_network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, residual_dim),
            )

        if concept_dim > 0 and residual_dim > 0:
            self.mi_estimator = CLUB(concept_dim, residual_dim, hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(concept_dim + residual_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        bottleneck = []

        concept_preds = None
        if self.concept_network is not None:
            concept_preds = self.concept_network(x)
            bottleneck.append(concept_preds.detach())

        residual = None
        if self.residual_network is not None:
            residual = self.residual_network(x)
            bottleneck.append(residual)

        bottleneck = torch.cat(bottleneck, dim=-1)
        output = self.predictor(bottleneck)
        return concept_preds, residual, output



# Initialize the model, loss function, and optimizer
model = ConceptModel(residual_dim=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

concept_loss_fn = nn.BCELoss()
predictor_loss_fn = nn.BCELoss()

# Training loop with tqdm progress bars
num_epochs = 2
losses = []

epoch_loop = tqdm(range(num_epochs), desc="Epochs")
#for epoch in tqdm(range(num_epochs), desc="Epochs"):
for epoch in epoch_loop:
    concept_losses = []
    predictor_losses = []
    epoch_losses = []

    for (X, concepts), y in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        concept_preds, residual, output = model(X)

        loss = 0

        predictor_loss = predictor_loss_fn(output.view(*y.shape), y.float())
        loss += predictor_loss

        if concept_preds is not None:
            concept_loss = concept_loss_fn(concept_preds, concepts)
            loss += concept_loss

        if residual is not None:
            pass

        loss.backward()
        optimizer.step()

        concept_losses.append(concept_loss.item())
        predictor_losses.append(predictor_loss.item())
        epoch_losses.append(loss.item())

    epoch_loop.set_description((
        f'epoch={epoch+1}, '
        f'concept_loss={np.mean(concept_losses):.4f}, '
        f'predictor_loss={np.mean(predictor_losses):.4f}'
    ))

# # Plot the training loss
# plt.plot(losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()

output_dim = 1
hidden_dim = 128
concept_dim = 2

new_predictor = nn.Sequential(
    nn.Linear(concept_dim, hidden_dim), nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
    nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
)
new_optimizer = optim.Adam(new_predictor.parameters(), lr=0.001)
new_loss_fn = nn.BCELoss()

# Training loop with tqdm progress bars
num_epochs = 10
losses = []

epoch_loop = tqdm(range(num_epochs), desc="Epochs")
for epoch in epoch_loop:
    epoch_losses = []

    for (X, concepts), y in tqdm(train_loader, leave=False):
        with torch.no_grad():
            concept_preds, _, _ = model(X)

        new_optimizer.zero_grad()

        output = new_predictor(concept_preds)
        print(output.mean().item(), output.std().item())
        loss = new_loss_fn(output.view(*y.shape), y.float())
        loss.backward()
        new_optimizer.step()

        epoch_losses.append(loss.item())

    epoch_loop.set_description((
        f'epoch={epoch+1}, '
        f'loss={np.mean(epoch_losses):.4f}'
    ))






# Test the model on the test dataset
model.eval()
new_predictor.eval()
correct = 0
total = 0

with torch.no_grad():
    for (X, concepts), y in test_loader:
        concept_preds, residual, output = model(X)
        output = new_predictor(concept_preds)
        predicted = (output > 0.5).long().view(*y.shape)
        #print(predicted.shape, y.shape, (predicted == y).shape)
        #_, predicted = torch.max(output.data, 1)
        total += y.numel()
        correct += (predicted == y).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test dataset: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'mnist_classification_model.pth')
