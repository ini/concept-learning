import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable



def train_multiclass_classification(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    num_epochs: int = 10,
    lr: float = 0.001,
    preprocess_fn: Callable = lambda batch: batch,
    callback_fn: Callable = lambda model, epoch, batch_index, batch: None,
    loss_fn: Callable = lambda data, output, target: F.cross_entropy(output, target),
    predict_fn: Callable = lambda output: output.argmax(dim=-1),
    save_path: str | None = None,
    save_interval: int | None = None,
):
    """
    Train a model for multiclass classification.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Train data
    test_loader : DataLoader
        Test data
    num_epochs : int
        Number of epochs to train for
    lr : float
        Learning rate
    callback_fn : Callable(model, epoch, batch_index, batch)
        Callback triggered before each training step on a batch
    preprocess_fn : Callable(batch) -> (X, y)
        Function to preprocess the batch before passing it to the model
    loss_fn : Callable(data, output, target) -> loss
        Loss function for the model
    predict_fn : Callable(output) -> prediction
        Function to convert the model's output to a prediction
    save_path : str
        Path to save the model to
    save_interval : int
        Epoch interval at which to save the model during training
    """
    # Train
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        epoch_losses = []
        batch_index = 0
        with tqdm(train_loader, desc='Batches', leave=False) as batches_loop:
            for batch in batches_loop:
                callback_fn(model, epoch, batch_index, batch)
                X, y = preprocess_fn(batch)

                # Update the model
                optimizer.zero_grad()
                output = model(X)
                loss = loss_fn(batch[0], output, y)
                loss.backward()
                optimizer.step()

                # Update the progress bar description with the loss
                epoch_losses.append(loss.item())
                batches_loop.set_postfix(loss=sum(epoch_losses) / len(epoch_losses))
                batch_index += 1

        if save_path and save_interval and (epoch % save_interval) == 0:
            torch.save(model.state_dict(), save_path)

    # Save
    if save_path:
        torch.save(model.state_dict(), save_path)

    # Test
    if test_loader is not None:
        model.eval()
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                X, y = preprocess_fn(batch)
                output = model(X)
                prediction = predict_fn(output)
                num_correct += (prediction == y).sum().item()
                num_samples += y.size(0)

        accuracy = 100 * num_correct / num_samples
        tqdm.write(f"Test Accuracy: {accuracy:.2f}%")

    return model

def concepts_preprocess_fn(batch):
    (X, concepts), y = batch
    return X, y

def cross_correlation(X: Tensor, Y: Tensor):
    """
    Compute the cross-correlation matrix between X and Y.

    Parameters
    ----------
    X : Tensor of shape (num_samples, x_dim)
        X samples
    Y : Tensor of shape (num_samples, y_dim)
        Y samples
    
    Returns
    -------
    R : Tensor of shape (x_dim, y_dim)
        Cross-correlation matrix
    """
    eps = torch.tensor(1e-6)
    X = (X - X.mean(dim=0)) / torch.maximum(X.std(dim=0), eps)
    Y = (Y - Y.mean(dim=0)) / torch.maximum(Y.std(dim=0), eps)
    return torch.bmm(X.unsqueeze(-1), Y.unsqueeze(1)).mean(dim=0)
