from __future__ import annotations

from lib.ccw import EYE
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from typing import Any, Callable, Iterable, Literal

from nn_extensions import VariableKwargs
from utils import accuracy, unwrap, zero_loss_fn, remove_prefix, remove_keys_with_prefix
from .concept_mixture import ConceptMixture
from nn_extensions import Chain

### Typing

ConceptBatch = tuple[tuple[Tensor, Tensor], Tensor]  # ((data, concepts), targets)


### Concept Models


class ConceptModel(nn.Module):
    """
    Base class for concept models.

    Attributes
    ----------
    base_network : nn.Module
        Network that pre-processes input
    concept_network : nn.Module
        Network that takes the output of `base_network` and
        generates concept logits
    residual_network : nn.Module
        Network that takes the output of `base_network` and
        generates a residual vector
    bottleneck_layer : nn.Module
        Network that post-processes the concatenated output of
        `concept_network` and `residual_network`
    target_network : nn.Module
        Network that takes the output of `bottleneck_layer` and
        generates target logits
    training_mode : one of {'independent', 'sequential', 'joint'}
        Training mode (see https://arxiv.org/abs/2007.04612)
    """

    def __init__(
        self,
        concept_network: nn.Module,
        residual_network: nn.Module,
        target_network: nn.Module,
        base_network: nn.Module = nn.Identity(),
        bottleneck_layer: nn.Module = nn.Identity(),
        concept_type: Literal["binary", "continuous"] = "binary",
        training_mode: Literal["independent", "sequential", "joint"] = "independent",
        mixer=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        concept_network : nn.Module -> (..., concept_dim)
            Concept prediction network
        residual_network : nn.Module -> (..., residual_dim)
            Residual network
        target_network : nn.Module (..., bottleneck_dim) -> (..., num_classes)
            Target network
        base_network : nn.Module
            Optional base network
        bottleneck_layer : nn.Module (..., bottleneck_dim) -> (..., bottleneck_dim)
            Optional bottleneck layer
        concept_type : one of {'binary', 'continuous'}
            Concept type
        training_mode : one of {'independent', 'sequential', 'joint'}
            Training mode (see https://arxiv.org/abs/2007.04612)
        """
        super().__init__()

        def freeze_layers_except_final(model, final_layer_name="fc"):
            """
            Freezes all layers except the specified final layer.

            Parameters
            ----------
            model : nn.Module
                The network containing layers to freeze.
            final_layer_name : str
                The name of the final layer to keep unfrozen.
            """
            # Iterate through named parameters in the base network
            for name, param in model.named_parameters():
                if (
                    final_layer_name not in name
                ):  # Freeze all layers not containing `final_layer_name`
                    param.requires_grad = False  # Freeze
                else:
                    param.requires_grad = True  # Keep final layer unfrozen
            return model

        if "freeze_backbone" in kwargs and kwargs["freeze_backbone"]:
            base_network = freeze_layers_except_final(base_network)

        self.base_network = VariableKwargs(base_network)

        self.concept_network = VariableKwargs(concept_network)
        self.residual_network = VariableKwargs(residual_network)
        self.target_network = VariableKwargs(target_network)
        self.bottleneck_layer = VariableKwargs(bottleneck_layer)
        self.mixer_model = VariableKwargs(mixer)

        self.concept_type = concept_type
        self.training_mode = training_mode
        self.ccm_mode = "" if "ccm_mode" not in kwargs else kwargs["ccm_mode"]
        self.ccm_network = (
            None
            if "ccm_network" not in kwargs
            else VariableKwargs(kwargs["ccm_network"])
        )
        if "base_model_ckpt" in kwargs:
            checkpoint_path = kwargs["base_model_ckpt"]
            state_dict = torch.load(checkpoint_path)["state_dict"]
            prefix_to_remove = "concept_model."
            modified_state_dict = remove_prefix(state_dict, prefix_to_remove)
            modified_state_dict = remove_keys_with_prefix(
                modified_state_dict, "concept_loss_fn"
            )
            modified_state_dict = remove_keys_with_prefix(
                modified_state_dict, "residual_loss_fn"
            )
            super().load_state_dict(modified_state_dict, strict=False)
        if self.training_mode == "semi_independent":
            concept_dim = kwargs["concept_dim"]
            residual_dim = kwargs["residual_dim"]
            self.mixer = ConceptMixture(
                concept_dim,
                residual_dim,
                self.target_network.module,
                self.mixer_model.module,
            )
            # freeze all parameters except self.mixer.mixer_model
            for name, param in self.named_parameters():
                if "mixer_model.module" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        else:
            self.mixer = None

    def forward(
        self,
        x: Tensor,
        concepts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values

        Returns
        -------
        concept_logits : Tensor
            Concept logits
        residual : Tensor
            Residual vector
        target_logits : Tensor
            Target logits
        """
        # Get concept logits & residual
        x = self.base_network(x, concepts=concepts)
        concept_logits = self.concept_network(x, concepts=concepts)

        if self.ccm_mode == "ccm_r" or self.ccm_mode == "ccm_eye":
            residual = self.residual_network(x.detach(), concepts=concepts)
        else:
            residual = self.residual_network(x, concepts=concepts)

        # Process concept logits & residual via bottleneck layer
        if not isinstance(unwrap(self.bottleneck_layer), nn.Identity):
            x = torch.cat([concept_logits, residual], dim=-1)
            x = self.bottleneck_layer(x, concepts=concepts)
            concept_dim, residual_dim = concept_logits.shape[-1], residual.shape[-1]
            concept_logits, residual = x.split([concept_dim, residual_dim], dim=-1)

        if self.ccm_mode == "" or self.ccm_mode == "ccm_eye":
            # Determine target network input based on training mode
            concept_preds = self.get_concept_predictions(concept_logits)
            if self.training and self.training_mode == "independent":
                x = torch.cat([concepts, residual], dim=-1)
                which = None
            elif self.training and self.training_mode == "sequential":
                x = torch.cat([concept_preds.detach(), residual], dim=-1)
                which = None
            elif self.training and self.training_mode == "semi_independent":
                # do selective interventions
                # start by choosing one or the other
                batch_size = x.shape[0]
                which = torch.rand(batch_size) > 0.5
                which = which.to(x.device)
                which = which.unsqueeze(1).expand(-1, concept_preds.shape[1])
                hard_maxed_concept_preds = concept_preds  # concept_preds > 0.5
                concept_joined = hard_maxed_concept_preds * which + concepts * (~which)
                x = torch.cat([concept_joined.detach(), residual], dim=-1)
            else:
                indices = torch.argmax(concept_preds, dim=-1)

                # Convert these indices to one-hot encoded form
                # hard_maxed_concept_preds = F.one_hot(
                #     indices, num_classes=concept_preds.shape[-1]
                # )
                hard_maxed_concept_preds = concept_preds  # concept_preds > 0.5
                x = torch.cat([hard_maxed_concept_preds, residual], dim=-1)
                if self.training_mode == "semi_independent":
                    which = (
                        torch.ones_like(hard_maxed_concept_preds).to(x.device).bool()
                    )
                else:
                    which = None

            # Get target logits
            if self.mixer is not None:
                target_logits = self.mixer(x, concepts=concepts, which=which)
            else:
                target_logits = self.target_network(x, concepts=concepts, which=which)
            return concept_logits, residual, target_logits
        else:
            if self.ccm_mode == "ccm_r":
                concept_preds = self.get_concept_predictions(concept_logits)
                target_one = self.target_network(concept_preds, concepts=concepts)
                target_two = self.ccm_network(residual, concepts=concepts)
                target_logits = target_one.detach() + target_two
                return concept_logits, residual, target_logits
            else:
                assert 0, f"{self.ccm_mode} not implemented yet"

    def get_concept_predictions(self, concept_logits: Tensor) -> Tensor:
        """
        Compute concept predictions from logits.
        """
        if self.concept_type == "binary":
            return concept_logits.sigmoid()

        return concept_logits


### Concept Models with PyTorch Lightning


class ConceptLightningModel(pl.LightningModule):
    """
    Lightning module for training a concept model.
    """

    def __init__(
        self,
        concept_model: ConceptModel,
        concept_loss_fn: Callable | None = F.binary_cross_entropy_with_logits,
        residual_loss_fn: Callable | None = None,
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        weight_decay=0,
        lr_step_size=None,
        lr_gamma=None,
        reg_type=None,
        reg_gamma: float = 1.0,
        lr_scheduler="cosine",
        chosen_optim="adam",
        **kwargs,
    ):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        concept_loss_fn : Callable(concept_logits, concepts) -> loss
            Concept loss function
        residual_loss_fn : Callable(residual, concepts) -> loss
            Residual loss function
        lr : float
            Learning rate
        alpha : float
            Weight for concept loss
        beta : float
            Weight for residual loss
        """
        if "concept_dim" in kwargs and kwargs["concept_dim"] == 0:
            concept_loss_fn = None
        if "residual_dim" in kwargs and kwargs["residual_dim"] == 0:
            residual_loss_fn = None

        super().__init__()
        self.concept_model = concept_model
        self.concept_loss_fn = concept_loss_fn or zero_loss_fn
        self.residual_loss_fn = residual_loss_fn or zero_loss_fn
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": True}

        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.reg_type = reg_type
        self.reg_gamma = reg_gamma
        self.lr_scheduler = lr_scheduler
        self.chosen_optim = chosen_optim

    def dummy_pass(self, loader: Iterable[ConceptBatch]):
        """
        Run a dummy forward pass to handle any uninitialized parameters.

        Parameters
        ----------
        loader : Iterable[ConceptBatch]
            Data loader
        """
        with torch.no_grad():
            (data, concepts), targets = next(iter(loader))
            self.concept_model(data, concepts=concepts)

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass.
        """
        return self.concept_model.forward(*args, **kwargs)

    def loss_fn(
        self,
        batch: ConceptBatch,
        outputs: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Compute loss.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        outputs : tuple[Tensor, Tensor, Tensor]
            Concept model outputs (concept_logits, residual, target_logits)
        """
        (data, concepts), targets = batch
        concept_logits, residual, target_logits = outputs

        # Concept loss
        concept_loss = self.concept_loss_fn(concept_logits, concepts)
        if concept_loss.requires_grad:
            self.log("concept_loss", concept_loss, **self.log_kwargs)

        # Residual loss
        residual_loss = self.residual_loss_fn(residual, concepts)
        if residual_loss.requires_grad:
            self.log("residual_loss", residual_loss, **self.log_kwargs)

        # Regularization loss
        if self.reg_type == "l1":
            if type(self.concept_model.target_network) == Chain:
                # If the target network is a Chain, the target network is the first module in the chain
                net_y = self.concept_model.target_network[1].module
            else:
                net_y = self.concept_model.target_network.module
            if not isinstance(net_y, nn.Linear):
                net_y = net_y[1]
            A = net_y.weight.abs().sum(0)

            def compute_l1_loss(w):
                return torch.abs(w).sum()

            reg_loss = compute_l1_loss(A)

        elif self.reg_type == "eye":
            if type(self.concept_model.target_network) == Chain:
                # If the target network is a Chain, the target network is the first module in the chain
                net_y = self.concept_model.target_network[1].module
            else:
                net_y = self.concept_model.target_network.module
            if not isinstance(net_y, nn.Linear):
                net_y = net_y[1]
            device = residual.device
            r = torch.cat(
                [torch.ones(concept_logits.shape[1]), torch.zeros(residual.shape[1])]
            ).to(device)
            reg_loss = EYE(r, net_y.weight.abs().mean(0))
        else:
            reg_loss = 0.0

        # Target loss
        target_loss = F.cross_entropy(target_logits, targets)
        if target_loss.requires_grad:
            self.log("target_loss", target_loss, **self.log_kwargs)

        return (
            target_loss
            + (self.alpha * concept_loss)
            + (self.beta * residual_loss)
            + (self.reg_gamma * reg_loss)
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        """

        if self.chosen_optim == "adam":
            optimizer = torch.optim.Adam(
                self.concept_model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.chosen_optim == "sgd":
            optimizer = torch.optim.SGD(
                self.concept_model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            assert 0, f"{self.chosen_optim} not implemented yet"

        if self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_step_size,
                gamma=self.lr_gamma,
            )
        elif self.lr_scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100,
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def step(
        self,
        batch: ConceptBatch,
        split: Literal["train", "val", "test"],
    ) -> Tensor:
        """
        Training / validation / test step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        split : one of {'train', 'val', 'test'}
            Dataset split
        """
        (data, concepts), targets = batch

        outputs = self.concept_model(data, concepts=concepts)
        concept_logits, residual, target_logits = outputs

        # Track loss
        loss = self.loss_fn(batch, outputs)
        self.log(f"{split}_loss", loss, **self.log_kwargs)

        # Track accuracy
        acc = accuracy(target_logits, targets)
        self.log(f"{split}_acc", acc, **self.log_kwargs)

        # Track concept accuracy
        if self.concept_model.concept_type == "binary":
            concept_acc = accuracy(concept_logits, concepts, task="binary")
            self.log(f"{split}_concept_acc", concept_acc, **self.log_kwargs)
        else:
            concept_rmse = F.mse_loss(concept_logits, concepts).sqrt()
            self.log(f"{split}_concept_rmse", concept_rmse, **self.log_kwargs)

        return loss

    def training_step(self, batch: ConceptBatch, batch_idx: int) -> Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        return self.step(batch, split="train")

    def validation_step(self, batch: ConceptBatch, batch_idx: int) -> Tensor:
        """
        ValConceptModelidation step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        return self.step(batch, split="val")

    def test_step(self, batch: ConceptBatch, batch_idx: int) -> Tensor:
        """
        Test step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        return self.step(batch, split="test")
