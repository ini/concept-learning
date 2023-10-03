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
import tempfile

if TYPE_CHECKING:
    from models import ConceptModel

from ray import train
import ray
import os
from ray.train import Checkpoint, ScalingConfig

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



def train_multiclass_classification_ray(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    preprocess_fn: Callable = lambda batch: batch,
    callback_fn: Callable = lambda model, epoch, batch_index, batch: None,
    loss_fn: Callable = lambda data, output, target: F.cross_entropy(output, target),
    config = None,
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
    num_epochs = config["num_epochs"]
    lr =  config["lr"]
    save_path = config["save_dir"]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    # Train the model
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        epoch_losses = []
        batch_index = 0
        with tqdm(train_loader, desc='Batches', leave=False) as batches_loop:
            for batch in batches_loop:
                callback_fn(model, epoch, batch_index, batch)
                X, y = preprocess_fn(batch)
                X = X.to(device)
                y = y.to(device)

                # Update the model
                optimizer.zero_grad()
                output = model(X)
                dat_cuda = [d.to(device) for d in batch[0]] if type(batch[0]) == list else batch[0].to(device)
                loss = loss_fn(dat_cuda, output, y)
                loss.backward()
                optimizer.step()

                # Update the progress bar description with the loss
                epoch_losses.append(loss.item())
                batches_loop.set_postfix(loss=sum(epoch_losses) / len(epoch_losses))
                batch_index += 1

        scheduler.step()

    

        metrics = {"loss": loss.item()}

        if epoch % config.get("checkpoint_freq", 1) == 0:
            test_acc = concept_model_accuracy(model, test_loader)
            metrics["test_acc"] = test_acc

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0

            # In standard DDP training, where the model is the same across all ranks,
            # only the global rank 0 worker needs to save and report the checkpoint
            print(train.get_context().get_world_rank())
            if should_checkpoint:
                torch.save(
                    model.state_dict(),  # NOTE: Unwrap the model.
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            print(checkpoint)
            train.report(metrics, checkpoint=checkpoint)

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


def get_mi_callback_fn_ray(
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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    mi_estimator = mi_estimator.to(device)
    def mi_training_step_(model: nn.Module, epoch, batch_index, batch):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        (X, concepts), y = batch
        X, y = X.to(device), y.to(device)
        model = model.to(device)
        mi_optimizer.zero_grad()
        with torch.no_grad():
            residual = model.residual_network(X)
            concept_preds = model.concept_network(X)
        mi_loss = mi_estimator.learning_loss(residual, concept_preds)
        mi_loss.backward()
        mi_optimizer.step()

    return mi_training_step_





from collections.abc import MutableMapping
import functools
import copy
import collections
from ray import tune
import scipy.stats


# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def grid_search_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = grid_search_update(d.get(k, {}), v)
        else:
            d[k] = tune.grid_search(v)
    return d


############# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    obj_in_question = (
        (obj[pre] if type(obj) == dict else rgetattr(obj, pre)) if pre else obj
    )
    if type(obj_in_question) is dict:
        obj_in_question[post] = val
        return obj_in_question
    else:
        return setattr(obj_in_question, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if type(obj) == dict:
            if attr not in obj:
                obj[attr] = {}
            return obj[attr]
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


##############


############## https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


#######


def merge_dicts(d1, d2):
    """
    Args:
        d1 (dict): Dict 1.
        d2 (dict): Dict 2.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged


def deep_update(
    original,
    new_dict,
    new_keys_allowed=False,
    allow_new_subkey_list=None,
    override_all_if_type_changes=None,
):
    """Updates original dict with values from new_dict recursively.

    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the allow_new_subkey_list, then new subkeys can be introduced.

    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        allow_new_subkey_list (Optional[List[str]]): List of keys that
            correspond to dict values where new subkeys can be introduced.
            This is only at the top level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), iff the "type" key in that value dict changes.
    """
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if (
                k in override_all_if_type_changes
                and "type" in value
                and "type" in original[k]
                and value["type"] != original[k]["type"]
            ):
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


