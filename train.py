from __future__ import annotations
from loader import get_data_loaders

import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Callable

from club import CLUB
from models import ConceptModel, ConceptBottleneckModel, ConceptWhiteningModel
from utils import (
    concept_model_accuracy,
    cross_correlation,
    get_cw_callback_fn,
    get_mi_callback_fn,
    get_mi_callback_fn_ray,
    train_multiclass_classification,
    train_multiclass_classification_ray,
)



def train_bottleneck_joint(
    model: ConceptBottleneckModel,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    residual_loss_fn: Callable = lambda r, c: torch.tensor(0),
    alpha: float = 1.0,
    beta: float = 1.0,
    **kwargs) -> ConceptBottleneckModel:
    """
    Joint training of a concept bottleneck model.

    Parameters
    ----------
    model : ConceptBottleneckModel
        Model to train
    train_loader : DataLoader
        Train data loader
    test_loader : DataLoader
        Test data loader
    residual_loss_fn : Callable(residual, concept_preds) -> Tensor
        Function to compute the residual loss
    alpha : float
        Weight of the concept loss
    beta : float
        Weight of the residual loss
    **kwargs
        Additional arguments to pass to `train_multiclass_classification()`
    """
    def loss_fn(data, output, target):
        _, concepts = data
        concept_preds, residual, target_preds = output
        concept_target = concepts[..., :concept_preds.shape[-1]]
        concept_loss = nn.BCELoss()(concept_preds, concept_target)
        residual_loss = residual_loss_fn(residual, concept_preds)
        target_loss = nn.CrossEntropyLoss()(target_preds, target)
        return (alpha * concept_loss) + (beta * residual_loss) + target_loss

    train_multiclass_classification(
        model,
        train_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]),
        loss_fn=loss_fn,
        **kwargs,
    )

    if test_loader is not None:
        print(
            'Test Classification Accuracy:', concept_model_accuracy(model, test_loader))

def train_whitening(
    model: ConceptWhiteningModel,
    concept_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    alignment_frequency: int = 20,
    **kwargs):
    """
    Train a concept whitening model.

    Parameters
    ----------
    model : ConceptWhiteningModel
        Model to train
    concept_dim : int
        Number of concepts
    train_loader : DataLoader
        Train data loader
    test_loader : DataLoader
        Test data loader
    alignment_frequency : int
        Frequency at which to align the concept whitening layer (i.e. every N batches)
    **kwargs
        Additional arguments to pass to `train_multiclass_classification()`
    """
    train_multiclass_classification(
        model, train_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]),
        callback_fn=get_cw_callback_fn(
            train_loader, concept_dim, alignment_frequency=alignment_frequency),
        **kwargs,
    )

    if test_loader is not None:
        print(
            'Test Classification Accuracy:', concept_model_accuracy(model, test_loader))

def train(
    make_bottleneck_model_fn: Callable,
    make_whitening_model_fn: Callable,
    concept_dim: int,
    residual_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    save_dir: str | Path = './saved_models',
    bottleneck_alpha: float = 1.0,
    bottleneck_beta: float = 1.0,
    mi_estimator_hidden_dim: int = 256,
    mi_optimizer_lr: float = 0.001,
    whitening_alignment_frequency : int = 20,
    **kwargs) -> list[nn.Module]:
    """
    Trains the following models:
        * Concept bottleneck model without residual
        * Concept bottleneck model with latent residual
        * Concept bottleneck model with decorrelated residual
        * Concept bottleneck model with mutual information minimizing residual
        * Concept whitening model with concept-whitened residual

    Parameters
    ----------
    make_bottleneck_model_fn : Callable(residual_dim) -> ConceptBottleneckModel
        Function to create a concept bottleneck model
    make_whitening_model_fn : Callable(residual_dim) -> ConceptWhiteningModel
        Function to create a concept whitening model
    concept_dim : int
        Number of concepts
    residual_dim : int
        Dimension of the residual
    train_loader : DataLoader
        Train data loader
    test_loader : DataLoader
        Test data loader
    save_dir : str | Path
        Directory to save the trained models
    bottleneck_alpha : float
        Weight of the concept loss for bottleneck models
    bottleneck_beta : float
        Weight of the residual loss for bottleneck models
    mi_estimator_hidden_dim : int
        Hidden dimension of the mutual information estimator
    mi_optimizer_lr : float
        Learning rate of the mutual information optimizer
    whitening_alignment_frequency : int
        Frequency at which to align the concept whitening layer (i.e. every N batches)
    **kwargs
        Additional arguments to pass to `train_multiclass_classification()`
    """
    trained_models = []

    # Create save directory
    dataset_name = train_loader.dataset.__class__.__name__
    date = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
    save_dir = Path(save_dir).resolve() / dataset_name / date
    save_dir.mkdir(parents=True, exist_ok=True)
    print('Saving models to:', str(save_dir))

    # Train without residual
    model = make_bottleneck_model_fn(residual_dim=0)
    train_bottleneck_joint(
        model, train_loader,
        test_loader=test_loader,
        alpha=bottleneck_alpha,
        beta=bottleneck_beta,
        save_path=save_dir / 'no_residual.pt',
        **kwargs,
    )
    trained_models.append(model)

    # Train with latent residual
    model = make_bottleneck_model_fn(residual_dim=residual_dim)
    train_bottleneck_joint(
        model, train_loader,
        test_loader=test_loader,
        alpha=bottleneck_alpha,
        beta=bottleneck_beta,
        save_path=save_dir / 'latent_residual.pt',
        **kwargs,
    )
    trained_models.append(model)

    # With decorrelated residual
    model = make_bottleneck_model_fn(residual_dim=residual_dim)
    train_bottleneck_joint(
        model, train_loader,
        test_loader=test_loader,
        residual_loss_fn=lambda r, c: cross_correlation(r, c).square().mean(),
        alpha=bottleneck_alpha,
        beta=bottleneck_beta,
        save_path=save_dir / 'decorrelated_residual.pt',
        **kwargs,
    )
    trained_models.append(model)

    # With MI-minimized residual
    model = make_bottleneck_model_fn(residual_dim=residual_dim)
    device = next(model.parameters()).device
    mi_estimator = CLUB(residual_dim, concept_dim, mi_estimator_hidden_dim).to(device)
    mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=mi_optimizer_lr)
    train_bottleneck_joint(
        model, train_loader,
        test_loader=test_loader,
        residual_loss_fn=mi_estimator.forward,
        callback_fn=get_mi_callback_fn(mi_estimator, mi_optimizer),
        alpha=bottleneck_alpha,
        beta=bottleneck_beta,
        save_path=save_dir / 'mi_residual.pt',
        **kwargs,
    )
    trained_models.append(model)

    # With concept-whitened residual
    model = make_whitening_model_fn(residual_dim=residual_dim)
    train_whitening(
        model, concept_dim, train_loader,
        test_loader=test_loader,
        save_path=save_dir / 'whitened_residual.pt',
        alignment_frequency=whitening_alignment_frequency,
        **kwargs,
    )
    trained_models.append(model)

    open(save_dir / '.done', 'a').close()
    return trained_models

def load_models(
    load_dir: str | Path,
    make_bottleneck_model_fn: Callable[[int], ConceptBottleneckModel],
    make_whitening_model_fn: Callable[[int], ConceptWhiteningModel],
    residual_dim: int) -> dict[str, ConceptModel]:
    """
    Load saved models from the given directory.

    Returns
    -------
    models : dict[str, ConceptModel]
        Dictionary of saved models
            * Concept bottleneck model without residual
            * Concept bottleneck model with latent residual
            * Concept bottleneck model with decorrelated residual
            * Concept bottleneck model with mutual information minimizing residual
            * Concept whitening model with concept-whitened residual
    """
    load_dir = Path(load_dir)
    models = {}

    models['no_residual'] = make_bottleneck_model_fn(residual_dim=0)
    models['no_residual'].load_state_dict(
        torch.load(load_dir / 'no_residual.pt'))

    models['latent_residual'] = make_bottleneck_model_fn(residual_dim=residual_dim)
    models['latent_residual'].load_state_dict(
        torch.load(load_dir / 'latent_residual.pt'))

    models['decorrelated_residual'] = make_bottleneck_model_fn(residual_dim=residual_dim)
    models['decorrelated_residual'].load_state_dict(
        torch.load(load_dir / 'decorrelated_residual.pt'))
    
    models['mi_residual'] = make_bottleneck_model_fn(residual_dim=residual_dim)
    models['mi_residual'].load_state_dict(
        torch.load(load_dir / 'mi_residual.pt'))
    
    models['whitened_residual'] = make_whitening_model_fn(residual_dim=residual_dim)
    models['whitened_residual'].load_state_dict(
        torch.load(load_dir / 'whitened_residual.pt'))

    return models



def train_bottleneck_joint_ray(
    model: ConceptBottleneckModel,
    callback_fn: Callable = lambda model, epoch, batch_index, batch: None,
    residual_loss_fn: Callable = lambda r, c: torch.tensor(0),
    config=dict()) -> ConceptBottleneckModel:
    """
    Joint training of a concept bottleneck model.

    Parameters
    ----------
    model : ConceptBottleneckModel
        Model to train
    train_loader : DataLoader
        Train data loader
    test_loader : DataLoader
        Test data loader
    residual_loss_fn : Callable(residual, concept_preds) -> Tensor
        Function to compute the residual loss
    alpha : float
        Weight of the concept loss
    beta : float
        Weight of the residual loss
    **kwargs
        Additional arguments to pass to `train_multiclass_classification()`
    """
    train_loader, test_loader, _, _ = get_data_loaders(
       config["dataset_name"], batch_size=config["batch_size"], data_dir=config["data_dir"])
    alpha = config["alpha"]
    beta = config["beta"]
    def loss_fn(data, output, target):
        _, concepts = data
        concept_preds, residual, target_preds = output
        concept_target = concepts[..., :concept_preds.shape[-1]]
        concept_loss = nn.BCELoss()(concept_preds, concept_target)
        residual_loss = residual_loss_fn(residual, concept_preds)
        target_loss = nn.CrossEntropyLoss()(target_preds, target)
        return (alpha * concept_loss) + (beta * residual_loss) + target_loss

    train_multiclass_classification_ray(
        model,
        train_loader,
        test_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]),
        loss_fn=loss_fn,
        callback_fn=callback_fn,
        config=config
    )

    if test_loader is not None:
        print(
            'Test Classification Accuracy:', concept_model_accuracy(model, test_loader))

def train_whitening_ray(
    model: ConceptWhiteningModel,
    concept_dim: int,
    config=dict(),):
    """
    Train a concept whitening model.

    Parameters
    ----------
    model : ConceptWhiteningModel
        Model to train
    concept_dim : int
        Number of concepts
    train_loader : DataLoader
        Train data loader
    test_loader : DataLoader
        Test data loader
    alignment_frequency : int
        Frequency at which to align the concept whitening layer (i.e. every N batches)
    **kwargs
        Additional arguments to pass to `train_multiclass_classification()`
    """
    alignment_frequency = config["whitening_alignment_frequency"]
    train_loader, test_loader, _, _ = get_data_loaders(
       config["dataset_name"], batch_size=config["batch_size"], data_dir=config["data_dir"])
    train_multiclass_classification_ray(
        model,
        train_loader,
        test_loader,
        preprocess_fn=lambda batch: (batch[0][0], batch[1]),
        callback_fn=get_cw_callback_fn(
            train_loader, concept_dim, alignment_frequency=alignment_frequency),
    )

    # if test_loader is not None:
    #     print(
    #         'Test Classification Accuracy:', concept_model_accuracy(model, test_loader))



def train_ray(
    config : dict, 
    make_bottleneck_model_fn: Callable,
    make_whitening_model_fn: Callable,
    ):
    '''
    Train a model using Ray Tune.
    '''
    # Create save directory
    

    
    if config["model_type"] == "bottleneck":
        # Train without residual
        model = make_bottleneck_model_fn(residual_dim=0)
        train_bottleneck_joint_ray(
            model,
            config=config
        )
    elif config["model_type"] == "baseline":
        model = make_bottleneck_model_fn(residual_dim=config["residual_dim"])
        train_bottleneck_joint_ray(
            model,
            config=config
        )
    elif config["model_type"] == "corr":
        # With decorrelated residual
        model = make_bottleneck_model_fn(residual_dim=config["residual_dim"])
        train_bottleneck_joint_ray(
            model, 
            residual_loss_fn=lambda r, c: cross_correlation(r, c).square().mean(),
            config=config
        )
    elif config["model_type"] == "mi":
        # With MI-minimized residual
        model = make_bottleneck_model_fn(residual_dim=config["residual_dim"])
        device = next(model.parameters()).device
        mi_estimator = CLUB(config['residual_dim'], config["concept_dim"], config["mi_estimator_hidden_dim"]).to(device)
        mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=config["mi_optimizer_lr"])
        train_bottleneck_joint_ray(
            model,
            residual_loss_fn=mi_estimator.forward,
            callback_fn=get_mi_callback_fn_ray(mi_estimator, mi_optimizer),
            config=config
        )
    elif config["model_type"] == "whitening":
        # With concept-whitened residual
        model = make_whitening_model_fn(residual_dim=config["residual_dim"])
        train_whitening_ray(
            model, config["concept_dim"],
            config=config,
        )


