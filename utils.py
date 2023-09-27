import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any
from typing import Callable



def to_device(x: Any, device: torch.device | str):
    """
    Move a tensor or collection of tensors to a device.
    """
    if isinstance(x, Tensor):
        return x.to(device)
    elif isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x, list):
        return [to_device(xi, device) for xi in x]
    return x

def train_multiclass_classification(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.001,
    preprocess_fn: Callable = lambda batch: batch,
    callback_fn: Callable = lambda model, epoch, batch_index, batch: None,
    loss_fn: Callable = lambda data, output, target: F.cross_entropy(output, target),
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
    num_epochs : int
        Number of epochs to train for
    lr : float
        Learning rate
    preprocess_fn : Callable(batch) -> (X, y)
        Function to preprocess the batch before passing it to the model
    callback_fn : Callable(model, epoch, batch_index, batch)
        Callback triggered before each training step on a batch
    loss_fn : Callable(data, output, target) -> loss
        Loss function for the model
    save_path : str
        Path to save the model to
    save_interval : int
        Epoch interval at which to save the model during training
    """
    device = next(model.parameters()).device

    # Create save directory if it doesn't exist
    if save_path:
        Path(save_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    # Train the model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        epoch_losses = []
        batch_index = 0
        with tqdm(train_loader, desc='Batches', leave=False) as batches_loop:
            for batch in batches_loop:
                batch = to_device(batch, device)
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

        scheduler.step()
        if save_path and save_interval and (epoch % save_interval) == 0:
            torch.save(model.state_dict(), save_path)

    # Save the trained model
    if save_path:
        torch.save(model.state_dict(), save_path)

def accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    preprocess_fn: Callable = lambda batch: batch,
    predict_fn: Callable = lambda output: output.argmax(dim=-1)) -> float:
    """
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    data_loader : DataLoader
        Data to evaluate on
    preprocess_fn : Callable(batch) -> (X, y)
        Function to preprocess the batch before passing it to the model
    predict_fn : Callable(output) -> prediction
        Function to convert the model's output to a prediction
    """
    device = next(model.parameters()).device

    # Test
    model.eval()
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            X, y = preprocess_fn(to_device(batch, device))
            output = model(X)
            prediction = predict_fn(output)
            num_correct += (prediction == y).sum().item()
            num_samples += y.numel()

    return num_correct / num_samples

def concepts_preprocess_fn(
    batch: tuple[tuple[Tensor, Tensor], Tensor]) -> tuple[Tensor, Tensor]:
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

def get_cw_callback_fn(
    data_loader: DataLoader, concept_dim: int, alignment_frequency: int = 20):
    """
    Returns a callback function for aligning the concept whitening layer
    (i.e. the `callback_fn` argument in `train_multiclass_classification`).

    Parameters
    ----------
    data_loader : DataLoader
        Loader for concept alignment data
    concept_dim : int
        Number of concepts
    alignment_frequency : int
        Frequency at which to align the concept whitening layer (i.e. every N batches)
    """
    print('Creating concept loaders for whitening ...')
    concept_loaders = [
        torch.utils.data.DataLoader(
            dataset=[
                x for ((x, concepts), y) in data_loader.dataset
                if concepts[concept_index] == 1 # assumes binary concepts
            ],
            batch_size=64,
            shuffle=True,
        )
        for concept_index in range(concept_dim)
    ]

    def align_concepts(model, epoch, batch_index, batch):
        if (batch_index + 1) % alignment_frequency == 0:
            model.eval()
            with torch.no_grad():
                for concept_index, concept_loader in enumerate(concept_loaders):
                    model.bottleneck_layer.mode = concept_index
                    for X in concept_loader:
                        X.requires_grad = True
                        model(X)
                        break

                    model.bottleneck_layer.update_rotation_matrix(cuda=False)
                    model.bottleneck_layer.mode = -1

            model.train()

    return align_concepts

def get_mi_callback_fn(
    mi_estimator: nn.Module, mi_optimizer: optim.Optimizer) -> Callable:
    """
    Return a callback function for training the mutual information estimator
    (i.e. the `callback_fn` argument in `train_multiclass_classification`).

    Parameters
    ----------
    mi_estimator : nn.Module
        Mutual information estimator
    mi_optimizer : optim.Optimizer
        Mutual information optimizer
    """
    def mi_training_step(model: nn.Module, epoch, batch_index, batch):
        (X, concepts), y = batch
        mi_optimizer.zero_grad()
        with torch.no_grad():
            residual = model.residual_network(X)
            concept_preds = model.concept_network(X)

        mi_loss = mi_estimator.learning_loss(residual, concept_preds)
        mi_loss.backward()
        mi_optimizer.step()

    return mi_training_step
