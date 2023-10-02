from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from models import ConceptModel



class Random(nn.Module):
    """
    Replaces input data with random noise.
    """

    def __init__(
        self, random_fn=torch.randn_like, indices: slice | Sequence[int] = slice(None)):
        """
        Parameters
        ----------
        random_fn : Callable(Tensor) -> Tensor
            Function to generate random noise
        indices : slice or Sequence[int]
            Feature indices to replace with random noise
        """
        super().__init__()
        self.random_fn = random_fn
        self.indices = indices

    def forward(self, x: Tensor):
        x[..., self.indices] = self.random_fn(x[..., self.indices])
        return x


def to_device(
    data: Tensor | tuple[Tensor] | list[Tensor], device: torch.device | str) -> Any:
    """
    Move a tensor or collection of tensors to the given device.

    Parameters
    ----------
    data : Tensor or tuple[Tensor] or list[Tensor]
        Tensor or collection of tensors
    device : torch.device or str
        Device to move the tensor(s) to
    """
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]

    raise ValueError(f'Unsupported data type: {type(data)}')

def make_ffn(
    output_dim: int,
    hidden_dim: int = 256,
    num_hidden_layers: int = 2,
    flatten_input: bool = False,
    output_activation: nn.Module = nn.Identity()) -> nn.Module:
    """
    Create a feedforward network.

    Parameters
    ----------
    output_dim : int
        Dimension of the output
    hidden_dim : int
        Dimension of the hidden layers
    num_hidden_layers : int
        Number of hidden layers
    output_activation : nn.Module
        Activation function for the output layer
    """
    hidden_layers = []
    for _ in range(num_hidden_layers):
        hidden_layers.append(nn.LazyLinear(hidden_dim))
        hidden_layers.append(nn.ReLU())

    pre_input_layer = nn.Flatten() if flatten_input else nn.Identity()
    return nn.Sequential(
        pre_input_layer, *hidden_layers, nn.LazyLinear(output_dim), output_activation)

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

def concept_model_accuracy(model: 'ConceptModel', data_loader: DataLoader):
    """
    Compute accuracy for a concept model on the given data.

    Parameters
    ----------
    model : ConceptModel
        Model to evaluate
    data_loader : DataLoader
        Data to evaluate on
    """
    def predict_fn(outputs):
        if isinstance(outputs, tuple):
            return outputs[2].argmax(dim=-1)
        else:
            return outputs.argmax(dim=-1)

    return accuracy(
        model, data_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]),
        predict_fn=predict_fn,
    )

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
    try:
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
    except ValueError as e:
        print('Error creating concept loaders:', e)
        return lambda model, epoch, batch_index, batch: None

    def align_concepts(model, epoch, batch_index, batch):
        if (batch_index + 1) % alignment_frequency == 0:
            model.eval()
            with torch.no_grad():
                for concept_index, concept_loader in enumerate(concept_loaders):
                    model.bottleneck_layer.mode = concept_index
                    for X in concept_loader:
                        X.requires_grad = True
                        model(X.to(next(model.parameters()).device))
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
