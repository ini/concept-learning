import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Define a transform to preprocess the data (you can customize this)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the original MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Define a function to filter dataset by class
def filter_dataset_by_class(dataset, target_classes):
    target_classes = torch.tensor(list(target_classes))
    mask = torch.isin(dataset.targets, target_classes)
    indices = torch.arange(len(dataset))[mask]
    return Subset(dataset, indices)

# Create datasets for digits 4 and 5, and for all other digits
train_dataset_A = filter_dataset_by_class(train_dataset, {4, 5})
test_dataset_A = filter_dataset_by_class(test_dataset, {4, 5})
train_dataset_B = filter_dataset_by_class(train_dataset, set(range(10)) - {4, 5})
test_dataset_B = filter_dataset_by_class(test_dataset, set(range(10)) - {4, 5})

print(len(train_dataset), len(test_dataset))
print(len(train_dataset_A), len(train_dataset_B), len(test_dataset_A), len(test_dataset_B))

# Create DataLoader instances for both training and test sets
batch_size = 64
train_loader_A = DataLoader(train_dataset_A, batch_size=batch_size, shuffle=True)
test_loader_A = DataLoader(test_dataset_A, batch_size=batch_size, shuffle=False)
train_loader_B = DataLoader(train_dataset_B, batch_size=batch_size, shuffle=True)
test_loader_B = DataLoader(test_dataset_B, batch_size=batch_size, shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)  # Two classes: 4 and 5
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)  # Apply softmax for multiclass classification
        return x

# Initialize the model, loss function, and optimizer
model_A = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_A.parameters(), lr=0.0001)

# Training loop with tqdm progress bars
num_epochs = 10
losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    
    with tqdm(train_loader_A, desc=f"Epoch {epoch + 1}/{num_epochs}") as epoch_progress:
        for batch_X, batch_y in epoch_progress:
            optimizer.zero_grad()
            outputs = model_A(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            epoch_progress.set_postfix(loss=loss.item())  # Update the progress bar description with the loss
    
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    epoch_progress.set_postfix(loss=epoch_loss)
    losses.append(epoch_loss)

# Calculate and print the final accuracy
model_A.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader_A:
        outputs = model_A(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on the test dataset: {accuracy:.2f}%")

# Plot the training loss
# plt.plot(losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()

# # Save the trained model
# torch.save(model.state_dict(), 'mnist_4_5_classification_model.pth')






# Initialize the model, loss function, and optimizer
model_B = nn.Sequential(
    nn.Flatten(),
    nn.Linear(2, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_B.parameters(), lr=0.0001)

# Training loop with tqdm progress bars
num_epochs = 10
losses = []



for epoch in range(num_epochs):
    epoch_losses = []
    
    with tqdm(train_loader_B, desc=f"Epoch {epoch + 1}/{num_epochs}") as epoch_progress:
        for batch_X, batch_y in epoch_progress:
            with torch.no_grad():
                X = model_A(batch_X)[..., [4, 5]].softmax(dim=-1)

            optimizer.zero_grad()
            outputs = model_B(X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
            epoch_progress.set_postfix(loss=loss.item())  # Update the progress bar description with the loss
    
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(epoch_loss)

# Calculate and print the final accuracy
model_B.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader_B:
        outputs = model_B(model_A(batch_X)[..., [4, 5]].softmax(dim=-1))
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on the test dataset: {accuracy:.2f}%")
