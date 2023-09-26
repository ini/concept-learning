import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Define a transform to preprocess the data (you can customize this)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the original MNIST dataset
original_train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)
original_test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform, download=True)

# Define a function to filter dataset by class
def filter_dataset_by_class(dataset, target_classes):
    indices = [i for i in range(len(dataset)) if dataset.targets[i] in target_classes]
    return indices

# Define a function to create a new dataset with images and parity targets
def create_parity_dataset(original_dataset, target_classes):
    target_indices = filter_dataset_by_class(original_dataset, target_classes)
    dataset = []

    for idx in target_indices:
        image, target = original_dataset[idx]
        parity_target = 0 if target % 2 == 0 else 1
        dataset.append((image, parity_target))
    
    return dataset

# Define the target classes (4 and 5)
target_classes_A = [4, 5]

# Create dataset A
dataset_A_train = create_parity_dataset(original_train_dataset, target_classes_A)
dataset_A_test = create_parity_dataset(original_test_dataset, target_classes_A)

# Define the target classes (0-3, 6-9)
target_classes_B = list(set(range(10)) - set(target_classes_A))

# Create dataset B
dataset_B_train = create_parity_dataset(original_train_dataset, target_classes_B)
dataset_B_test = create_parity_dataset(original_test_dataset, target_classes_B)

# Create DataLoader instances for train and test sets of both datasets A and B
batch_size = 64
train_loader_A = DataLoader(dataset_A_train, batch_size=batch_size, shuffle=True)
test_loader_A = DataLoader(dataset_A_test, batch_size=batch_size, shuffle=False)

train_loader_B = DataLoader(dataset_B_train, batch_size=batch_size, shuffle=True)
test_loader_B = DataLoader(dataset_B_test, batch_size=batch_size, shuffle=False)

# Check the number of samples in train and test sets of each dataset
print(f"Dataset A - Train: {len(dataset_A_train)}, Test: {len(dataset_A_test)}")
print(f"Dataset B - Train: {len(dataset_B_train)}, Test: {len(dataset_B_test)}")



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm


# Initialize the model, loss function, and optimizer
model_A = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1), nn.Sigmoid(),
)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_A.parameters(), lr=0.001)

# Training loop with tqdm progress bars
num_epochs = 20
losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    
    with tqdm(train_loader_A, desc=f"Epoch {epoch + 1}/{num_epochs}") as epoch_progress:
        for batch_X, batch_y in epoch_progress:
            optimizer.zero_grad()
            outputs = model_A(batch_X)
            loss = criterion(outputs.flatten(), batch_y.float())
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
        predicted = (outputs > 0.5).long().flatten()
        total += batch_y.flatten().size(0)
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
    nn.Linear(1, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1), nn.Sigmoid(),
)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_B.parameters(), lr=0.001)

# Training loop with tqdm progress bars
num_epochs = 20
losses = []



for epoch in range(num_epochs):
    epoch_losses = []
    
    with tqdm(train_loader_B, desc=f"Epoch {epoch + 1}/{num_epochs}") as epoch_progress:
        for batch_X, batch_y in epoch_progress:
            with torch.no_grad():
                X = (model_A(batch_X) > 0.5).float()
                X = torch.randn_like(X)

            optimizer.zero_grad()
            outputs = model_B(X)
            loss = criterion(outputs.flatten(), batch_y.float())
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
        X = (model_A(batch_X) > 0.5).float()
        X = torch.randn_like(X)
        outputs = model_B(X)
        predicted = (outputs > 0.5).long().flatten()
        total += batch_y.flatten().size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on the test dataset: {accuracy:.2f}%")
