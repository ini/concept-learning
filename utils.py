from __future__ import annotations

import os
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torchmetrics import Accuracy
from typing import Callable



def accuracy(logits: Tensor, targets: Tensor, task: str = 'multiclass') -> Tensor:
    """
    Compute accuracy from logits and targets.

    Parameters
    ----------
    logits : Tensor
        Logits
    targets : Tensor
        Targets
    """
    if task == 'binary':
        preds = logits.sigmoid()
        accuracy_fn = Accuracy(task='binary').to(logits.device)
    else:
        preds = logits.argmax(dim=-1)
        accuracy_fn = Accuracy(
            task='multiclass', num_classes=logits.shape[-1]).to(logits.device)

    return accuracy_fn(preds, targets)

def random_log_uniform(low: float, high: float, size: int | tuple[int] = ()) -> Tensor:
    """
    Sample from a log-uniform distribution.

    Parameters
    ----------
    low : float
        Lower bound
    high : float
        Upper bound
    size : int | tuple[int]
        Size of the sample
    """
    low, high = torch.as_tensor(low).log(), torch.as_tensor(high).log()
    return torch.exp(torch.rand(size) * (high - low) + low)

def is_sigmoid(fn: Callable) -> bool:
    """
    Check if a function is a sigmoid function.
    """
    if isinstance(fn, nn.Sigmoid):
        return True

    return fn in (
        torch.sigmoid, torch.sigmoid_, F.sigmoid, Tensor.sigmoid, Tensor.sigmoid_)

def logit_fn(probs: Tensor):
    """
    Logit function (i.e. inverse sigmoid function).
    """
    eps = random_log_uniform(1e-12, 1e-6)
    return torch.log(probs + eps) - torch.log(1 - probs + eps)

def to_device(
    data: Tensor | tuple[Tensor] | list[Tensor],
    device: torch.device | str) -> Tensor | tuple[Tensor] | list[Tensor]:
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

def unwrap(model: nn.Module) -> nn.Module:
    """
    Unwrap a model.

    Parameters
    ----------
    model : nn.Module
        Model to unwrap
    """
    return getattr(model, 'module', model)

def make_mlp(
    output_dim: int,
    hidden_dim: int = 256,
    num_hidden_layers: int = 2,
    flatten_input: bool = False) -> nn.Module:
    """
    Create a multi-layer perceptron.

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
    return nn.Sequential(pre_input_layer, *hidden_layers, nn.LazyLinear(output_dim))

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

def set_cuda_visible_devices(available_memory_threshold: float):
    """
    Set CUDA_VISIBLE_DEVICES to the GPUs whose fraction of available memory
    is at least a given threshold.

    When running processes with fractional GPUs, set the threshold to
    the fraction of the GPU memory that is available to each worker.

    Parameters
    ----------
    available_memory_threshold : float in range [0, 1]
        Threshold for fraction of available GPU memory
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError_LibraryNotFound:
        return

    available_devices = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if memory_info.free / memory_info.total >= available_memory_threshold:
            available_devices.append(i)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_devices))
