from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal, Tuple
from .base import ConceptModel, ConceptLightningModel
from utils import accuracy, auroc, unwrap, zero_loss_fn, remove_prefix, remove_keys_with_prefix


class ProbabilisticConceptModel(ConceptModel):
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
        concept_rank_model: nn.Module = nn.Identity(),
        concept_type: Literal["binary", "continuous"] = "binary",
        training_mode: Literal[
            "independent", "sequential", "joint", "intervention_aware"
        ] = "independent",
        **kwargs,
    ):
        super().__init__(
            concept_network=concept_network,
            residual_network=residual_network,
            target_network=target_network,
            base_network=base_network,
            bottleneck_layer=bottleneck_layer,
            concept_rank_model=concept_rank_model,
            concept_type=concept_type,
            training_mode=training_mode,
            **kwargs,
        )
    def forward(
        self,
        x: Tensor,
        concepts: Tensor | None = None,
        intervention_idxs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values
        intervention_idxs : Tensor or None
            Indices of interventions

        Returns
        -------
        concept_logits : Tensor
            Concept logits
        residual : Tensor
            Residual vector
        target_logits : Tensor
            Target logits
        """
        # if negative intervention, invert accurate concepts
        if self.negative_intervention:
            concepts = 1 - concepts

        # Get concept logits & residual
        # x = self.base_network(x, concepts=concepts)
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        #     x = self.base_network(x)
        # breakpoint()
        if x.device.type == "cpu":
            model_float = self.base_network.to(torch.float32)
            x = model_float(x)

        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = self.base_network(x)

        if not isinstance(x, torch.Tensor):
            # Handle non-tensor case
            x = x.logits
        x = x.float()

        output = self.concept_network(x, concepts=concepts)
        concept_logits, concept_logvars = torch.split(output, output.shape[-1] // 2, dim=-1)
        #reparametrization trick
        if self.training:
            std = torch.exp(0.5 * concept_logvars)
            eps = torch.randn_like(std)
            concept_logits = concept_logits + eps * std

        output = self.residual_network(x, concepts=concepts)
        residual_logits, residual_logvars = torch.split(output, output.shape[-1] // 2, dim=-1)

        if self.training:
            std = torch.exp(0.5 * residual_logvars)
            eps = torch.randn_like(std)
            residual_logits = residual_logits + eps * std

        # Process concept logits & residual via bottleneck layer
        if not isinstance(unwrap(self.bottleneck_layer), nn.Identity):
            assert 0, "Not implemented"


        # interventions on concepts for intervention_aware training
        if intervention_idxs is None:
            intervention_idxs = torch.zeros_like(concepts)
        target_logits = self.calc_target_preds(
        (concept_logits, concept_logvars), (residual_logits, residual_logvars), concepts, intervention_idxs
        )
        return (concept_logits, concept_logvars), (residual_logits, residual_logvars), target_logits
        # # intervene on concepts
        # concept_preds = (
        #     concept_preds.detach() * (1 - intervention_idxs)
        #     + concepts * intervention_idxs
        # )
        # concept_uncertainty = (
        #     concept_uncertainty.detach() * (1 - intervention_idxs)
        #     + torch.ones_like(concept_uncertainty) * intervention_idxs
        # )

        # if self.training and self.training_mode == "independent":
        #     x_concepts = concepts
        # elif self.training and self.training_mode == "sequential":
        #     x_concepts = concept_preds.detach()
        # else:
        #     # fully joint training
        #     x_concepts = concept_preds
        # attended_residual = self.cross_attention(
        #     x_concepts, residual_logits, intervention_idxs.detach()
        # )
        # if not self.additive_residual:
        #     x = torch.cat([x_concepts.detach(), attended_residual], dim=-1)
        # else:
        #     assert 0, "Not implemented"
        # # Get target logits
        # target_logits = self.target_network(
        #     x, concepts=concepts, intervention_idxs=intervention_idxs.detach()
        # )
        # if target_logits.shape[-1] == 1:
        #     target_logits = target_logits.squeeze(-1)

        # return concept_logits, residual, target_logits

    def calc_concept_group(
        self,
        concept_logits: Tensor,
        residual: Tensor,
        concepts: Tensor,
        intervention_idxs: Tensor,
        train: bool = False,
    ) -> Tensor:
        """
        Compute concept group scores.
        """
        if type(concept_logits) == tuple:
            concept_logits, concept_logvars = concept_logits
        if type(residual) == tuple:
            residual, residual_logvars = residual

        concept_preds = self.get_concept_predictions(concept_logits)

        concept_preds = (
            concept_preds.detach() * (1 - intervention_idxs)
            + concepts * intervention_idxs
        )
        concept_uncertainty = 1 / (1 + torch.exp(-torch.exp(concept_logvars)))
        concept_uncertainty = (
            concept_uncertainty.detach() * (1 - intervention_idxs)
            + torch.zeros_like(concept_uncertainty) * intervention_idxs
        )

        attended_residual = self.cross_attention(
            concept_preds.detach(), residual, intervention_idxs.detach()
        )
        residual_uncertainty = 1 / (1 + torch.exp(-torch.exp(residual_logvars)))
        if not self.additive_residual:
            x = torch.cat([concept_preds.detach(), concept_uncertainty.detach(), attended_residual, residual_uncertainty.detach()], dim=-1)
        else:
            x = concept_preds.detach() + attended_residual

        # rank_input = torch.concat(
        #     [x, intervention_idxs],
        #     dim=-1,
        # ).detach()
        rank_input = x.detach()

        next_concept_group_scores = self.concept_rank_model(rank_input)
        if train:
            return next_concept_group_scores

        # zero out the scores of the concepts that have already been intervened on
        next_concept_group_scores = torch.where(
            intervention_idxs == 1,
            torch.ones(intervention_idxs.shape).to(intervention_idxs.device) * (-1000),
            next_concept_group_scores,
        )
        # return the softmax of the scores
        return torch.nn.functional.softmax(
            next_concept_group_scores,
            dim=-1,
        )
    
    def calc_target_preds(
        self,
        concept_logits: Tensor,
        residual: Tensor,
        concepts: Tensor,
        intervention_idxs: Tensor,
        train: bool = False,
    ) -> Tensor:
        """
        Compute concept group scores.
        """
        if type(concept_logits) == tuple:
            concept_logits, concept_logvars = concept_logits
        if type(residual) == tuple:
            residual, residual_logvars = residual
        
        concept_preds = self.get_concept_predictions(concept_logits)
        concept_preds = (
            concept_preds.detach() * (1 - intervention_idxs)
            + concepts * intervention_idxs
        )
        concept_uncertainty = 1 / (1 + torch.exp(-torch.exp(concept_logvars)))
        concept_uncertainty = (
            concept_uncertainty.detach() * (1 - intervention_idxs)
            + torch.zeros_like(concept_uncertainty) * intervention_idxs
        )

        attended_residual = self.cross_attention(
            concept_preds.detach(), residual, intervention_idxs.detach()
        )
        residual_uncertainty = 1 / (1 + torch.exp(-torch.exp(residual_logvars)))
        if not self.additive_residual:
            x = torch.cat([concept_preds.detach(),concept_uncertainty.detach(), attended_residual, residual_uncertainty.detach()], dim=-1)
        else:
            x = concept_preds.detach() + attended_residual

        target_logits = self.target_network(
            x, concepts=concepts, intervention_idxs=intervention_idxs.detach()
        )
        if target_logits.shape[-1] == 1:
            target_logits = target_logits.squeeze(-1)
        return target_logits
