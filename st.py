from club import CLUB
from data_st import get_data_loaders

train_loader, test_loader, train_loader_A, test_loader_A, train_loader_B, test_loader_B = get_data_loaders()



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm



class ConceptModel(nn.Module):

    def __init__(
        self,
        input_dim=28*28, output_dim=10, hidden_dim=128,
        concept_dim=1, residual_dim=8,
    ):
        super().__init__()
        self.concept_dim, self.residual_dim = concept_dim, residual_dim

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
            nn.Linear(hidden_dim, output_dim),
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
    
    def concept_predictions(self, x):
        return self.concept_network(x)


import torch.nn.functional as F
from utils import train_multiclass_classification, concepts_preprocess_fn



model = ConceptModel(concept_dim=10, residual_dim=8, hidden_dim=128)
mi_estimator = CLUB(model.residual_dim, model.concept_dim, 128)
mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=0.001)

def loss_fn(data, output, target):
    _, concepts = data
    concept_preds, residual, output = output
    concept_loss = F.binary_cross_entropy(concept_preds, concepts)
    residual_loss = mi_estimator(residual, concept_preds)
    prediction_loss = F.cross_entropy(output, target)
    return concept_loss + residual_loss + prediction_loss

def mi_training_step(model, epoch, batch_index, batch):
    (X, concepts), y = batch
    mi_optimizer.zero_grad()
    with torch.no_grad():
        residual = model.residual_network(X)
        concept_preds = model.concept_network(X)

    mi_loss = mi_estimator.learning_loss(residual, concept_preds)
    mi_loss.backward()
    mi_optimizer.step()

model = train_multiclass_classification(
    model,
    train_loader,
    num_epochs=2,
    preprocess_fn=concepts_preprocess_fn,
    callback_fn=mi_training_step,
    loss_fn=loss_fn,
)


    #         mi_losses.append(mi_loss.item())
    #         concept_losses.append(concept_loss.item())
    #         residual_losses.append(residual_loss.item())
    #         prediction_losses.append(prediction_loss.item())
    #         epoch_losses.append(loss.item())

    #         # Update the progress bar description with the loss
    #         epoch_progress.set_postfix(
    #             mi_loss=mi_loss.item(),
    #             c_loss=concept_loss.item(),
    #             r_loss=residual_loss.item(),
    #             p_loss=prediction_loss.item(),
    #         )

    # epoch_loss = sum(epoch_losses) / len(epoch_losses)
    # epoch_progress.set_postfix(
    #     mi_loss=sum(mi_losses) / len(mi_losses),
    #     c_loss=sum(concept_losses) / len(concept_losses),
    #     r_loss=sum(residual_losses) / len(residual_losses),
    #     p_loss=sum(prediction_losses) / len(prediction_losses),
    # )
    # losses.append(epoch_loss)

model.eval()
correct = 0
total = 0

# Calculate and print the final accuracy
with torch.no_grad():
    for (X, concepts), y in test_loader:
        _, _, outputs = model(X)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on the test dataset: {accuracy:.2f}%")

with torch.no_grad():
    for (X, concepts), y in test_loader:
        concept_preds, _, _ = model(X)
        concept_preds = (concept_preds > 0.5).long().flatten()
        concepts = (concepts > 0.5).long().flatten()
        total += concepts.size(0)
        correct += (concepts == concept_preds).sum().item()

accuracy = 100 * correct / total
print(f"Final Concept Accuracy on the test dataset: {accuracy:.2f}%")

# Plot the training loss
# plt.plot(losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()

# # Save the trained model
# torch.save(model.state_dict(), 'mnist_4_5_classification_model.pth')




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







# Initialize the model, loss function, and optimizer
model_B = nn.Sequential(
    nn.Flatten(),
    nn.Linear(model.concept_dim, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 10), nn.Sigmoid(),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_B.parameters(), lr=0.001)

# Training loop with tqdm progress bars
num_epochs = 20
losses = []

for epoch in range(num_epochs):
    epoch_losses = []

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as epoch_progress:
        for (X, concepts), y in epoch_progress:
            with torch.no_grad():
                concept_preds, residual, output = model(X)

            optimizer.zero_grad()
            outputs = model_B(concept_preds)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # Update the progress bar description with the loss
            epoch_progress.set_postfix(loss=loss.item())
    
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(epoch_loss)

# Calculate and print the final accuracy
model_B.eval()
correct = 0
total = 0

with torch.no_grad():
    for (X, concepts), y in test_loader:
        concept_preds, residual, output = model(X)
        outputs = model_B(concept_preds)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on the test dataset: {accuracy:.2f}%")





# Initialize the model, loss function, and optimizer
model_C = nn.Sequential(
    nn.Flatten(),
    nn.Linear(model.residual_dim, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 10), nn.Sigmoid(),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_C.parameters(), lr=0.001)

# Training loop with tqdm progress bars
num_epochs = 20
losses = []

for epoch in range(num_epochs):
    epoch_losses = []

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as epoch_progress:
        for (X, concepts), y in epoch_progress:
            with torch.no_grad():
                concept_preds, residual, output = model(X)

            optimizer.zero_grad()
            outputs = model_C(residual)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # Update the progress bar description with the loss
            epoch_progress.set_postfix(loss=loss.item())
    
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(epoch_loss)

# Calculate and print the final accuracy
model_C.eval()
correct = 0
total = 0

with torch.no_grad():
    for (X, concepts), y in test_loader:
        concept_preds, residual, output = model(X)
        outputs = model_C(residual)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on the test dataset: {accuracy:.2f}%")
