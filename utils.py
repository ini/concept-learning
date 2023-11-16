from __future__ import annotations

import os
import pynvml
import torch
import torch.nn as nn

from collections import ChainMap
from ray.tune.search.variant_generator import generate_variants, grid_search
from torch import Tensor
from torchmetrics import Accuracy
from typing import Any



### Torch

def zero_loss_fn(*tensors: Tensor):
    """
    Dummy loss function that returns zero.
    """
    return torch.tensor(0.0, device=tensors[0].device)

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

def make_cnn(output_dim: int, cnn_type: str = 'resnet18') -> nn.Module:
    """
    Create a convolutional neural network.

    Parameters
    ----------
    output_dim : int
        Dimension of the output
    cnn_type : str
        CNN architecture
    """
    if cnn_type == 'resnet18':
        from torchvision.models.resnet import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    elif cnn_type == 'inception_v3':
        from torchvision.models.inception import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        model.aux_logits = False
        return model

    raise ValueError(f"Unknown CNN type: {cnn_type}")

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



### CUDA

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



### Ray

def process_grid_search_tuples(config: dict):
    """
    Process configuration dictionary with grid search tuples.

    Parameters
    ----------
    config : dict
        Configuration dictionary, with entries of the form:
            * k: v
            * (k_0, k_1, ..., k_n): grid_search([
                (v0_0, v0_1, ..., v0_n),
                (v1_0, v1_1, ..., v1_n),
                (v2_0, v2_1, ..., v2_n),
                ...,
            ])

    Example
    -------
    ```
    config = {
        ('a', 'b'): grid_search([(1, 2), (3, 4), (5, 6)]),
        'c': 7,
    }
    ```
    results in the following combinations:
        * `{'a': 1, 'b': 2, 'c': 7}`
        * `{'a': 3, 'b': 4, 'c': 7}`
        * `{'a': 5, 'b': 6, 'c': 7}`
    """
    # Turn all keys into tuples, and all values into a grid search over tuples
    config = {
        k if isinstance(k, tuple) else (k,): v if isinstance(k, tuple) else (v,)
        for k, v in config.items()
    }

    # Convert into a grid search over individual config dictionaries
    merge_dicts = lambda dicts: dict(ChainMap(*dicts))
    config = grid_search([
        merge_dicts(dict(zip(k, v)) for k, v in reversed(spec.items()))
        for _, spec in generate_variants(config)
    ])

    return config


class RayConfig(dict):
    """
    Handles get / set in config dictionaries with grid search.
    """

    def get(self, key: str, default: Any = ...) -> Any:
        """
        Get a value from the config.

        Parameters
        ----------
        key : str
            Config key
        default : Any
            Default value if key is not found
        """
        try:
            if key in self:
                value = self[key]
                if isinstance(value, dict) and len(value) == 1 and 'grid_search' in value:
                    return value['grid_search']
                else:
                    return value

            elif 'train_loop_config' in self:
                return self.get(self['train_loop_config'], key, default=default)

            elif 'grid_search' in self:
                values = {item[key] for item in self['grid_search']}
                assert len(values) == 1, f'Inconsistent values for {key}: {values}'
                return next(iter(values))

            raise KeyError

        except KeyError:

            if default is not ...:
                return default

        raise KeyError(f'Key not found: {key}')

    def set(self, key: str, value: Any):
        """
        Set a value in the config.

        Parameters
        ----------
        key : str
            Config key
        value : Any
            Config value
        """
        if 'grid_search' in self:
            for item in self['grid_search']:
                item[key] = value
        else:
            self[key] = value

    def update(self, other: dict):
        """
        Update the config from another dictionary.
        """
        for key, value in other.items():
            self.set(key, value)
