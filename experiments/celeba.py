import os
import ray
import torch.nn as nn

from models import ConceptModel, ConceptEmbeddingModel, make_bottleneck_layer
from nn_extensions import Apply
from utils import (
    make_cnn,
    process_grid_search_tuples,
    make_mlp,
    make_concept_embedding_model,
)
import torch
import torch.nn as nn


class CrossAttentionModel(nn.Module):
    def __init__(self, input_dim_c, input_dim_r, embed_dim, num_heads):
        super(CrossAttentionModel, self).__init__()

        # Embedding layers
        self.concept_embedding = nn.Linear(input_dim_c, embed_dim)
        self.concept_embedding_intervention = nn.Linear(input_dim_c, embed_dim)
        self.residual_embedding = nn.Linear(input_dim_r, embed_dim)

        # Cross-Attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Scaling factor for attended residuals
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, concepts, residuals, intervention_idxs, attention_mask=None):
        """
        concepts: Tensor of shape (batch_size, input_dim_c)
        residuals: Tensor of shape (batch_size, input_dim_r)
        attention_mask: Optional tensor for masking attention (batch_size, input_dim_c)
        """
        # Embed concepts and residuals
        non_intervene = concepts.detach() * (~(intervention_idxs.bool()))
        intervene = concepts.detach() * intervention_idxs
        concepts_embedded1 = self.concept_embedding(non_intervene)
        concepts_embedded_intervention = self.concept_embedding_intervention(intervene)
        concepts_embedded = concepts_embedded1 + concepts_embedded_intervention
        residuals_embedded = self.residual_embedding(
            residuals
        )  # (batch_size, embed_dim)

        # Normalize embeddings
        concepts_norm = self.layer_norm(concepts_embedded)  # (batch_size, embed_dim)
        residuals_norm = self.layer_norm(residuals_embedded)  # (batch_size, embed_dim)

        # Reshape for MultiheadAttention: (1, batch_size, embed_dim)
        concepts_norm = concepts_norm.unsqueeze(0)  # (1, batch_size, embed_dim)
        residuals_norm = residuals_norm.unsqueeze(0)  # (1, batch_size, embed_dim)

        # Apply cross-attention: residuals as queries, concepts as keys and values
        attended_residuals, _ = self.cross_attention(
            query=residuals_norm,  # (1, batch_size, embed_dim)
            key=concepts_norm,  # (1, batch_size, embed_dim)
            value=concepts_norm,  # (1, batch_size, embed_dim)
            key_padding_mask=attention_mask,  # (batch_size, input_dim_c) if provided
        )  # (1, batch_size, embed_dim)

        # Remove the sequence dimension
        attended_residuals = attended_residuals.squeeze(0)  # (batch_size, embed_dim)

        # Combine attended residuals with original residuals
        combined_residuals = (
            residuals_norm.squeeze(0) + self.scale * attended_residuals
        )  # (batch_size, embed_dim)

        return combined_residuals


class PassThrough(nn.Module):
    def __init__(self, input_dim_c, input_dim_r, embed_dim, num_heads):
        super(PassThrough, self).__init__()

    def forward(self, concepts, residuals, intervention_idxs, attention_mask=None):
        return residuals


def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config["num_classes"]
    concept_dim = config["concept_dim"]
    residual_dim = config["residual_dim"]
    training_mode = config.get("training_mode", "independent")
    int_model_use_bn = config.get("int_model_use_bn", True)
    int_model_layers = config.get("int_model_layers", None)
    backbone = config.get("backbone", "resnet34")

    num_hidden = config.get("num_hidden", 0)

    if config.get("model_type") == "cem":
        bottleneck_dim = (
            concept_dim * residual_dim
        )  # residual dim is the size of the concept embedding for cem
    else:
        bottleneck_dim = concept_dim + residual_dim

    if num_hidden > 0:
        target_network = make_mlp(
            num_classes,
            num_hidden_layers=num_hidden,
            hidden_dim=16,
            add_layer_norm=True,
        )
    else:
        target_network = nn.Linear(bottleneck_dim, num_classes)
        # nn.Linear(bottleneck_dim, num_classes)

    if config.get("model_type") == "cem":
        units = (
            [
                concept_dim * residual_dim + concept_dim
            ]  # for cem, input is concept_dim * residual_dim (# of concepts * concept embedding dim)
            + (int_model_layers or [256, 128])  # + previous interventions
            + [concept_dim]
        )
    else:
        units = (
            [
                concept_dim + residual_dim + concept_dim
            ]  # Bottleneck  # Prev interventions
            + (int_model_layers or [256, 128])
            + [concept_dim]
        )
    layers = []
    for i in range(1, len(units)):
        if int_model_use_bn:
            layers.append(
                torch.nn.BatchNorm1d(num_features=units[i - 1]),
            )
        layers.append(torch.nn.Linear(units[i - 1], units[i]))
        if i != len(units) - 1:
            layers.append(torch.nn.LeakyReLU())
    if config.get("intervention_weight", 0.0) > 0:
        concept_rank_model = torch.nn.Sequential(*layers)
    else:
        concept_rank_model = nn.Identity()

    if config.get("model_type") == "cem" or config.get("model_type") == "cem_mi":
        concept_prob_generators, concept_context_generators = (
            make_concept_embedding_model(
                1000, residual_dim, concept_dim, embedding_activation="leakyrelu"
            )
        )
        return ConceptEmbeddingModel(
            base_network=make_cnn(1000, cnn_type=backbone),
            concept_network=concept_prob_generators,
            residual_network=concept_context_generators,
            target_network=target_network,
            bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
            concept_rank_model=concept_rank_model,
            **config,
        )
    else:
        # return ConceptModel(
        #     base_network=make_cnn(1000, cnn_type="resnet34"),
        #     concept_network=nn.Linear(1000, concept_dim),
        #     residual_network=torch.nn.Sequential(
        #         *[
        #             torch.nn.Linear(
        #                 1000,
        #                 residual_dim,
        #             ),
        #             torch.nn.LeakyReLU(),
        #         ]
        #     ),
        #     target_network=target_network,
        #     bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
        #     concept_rank_model=concept_rank_model,
        #     **config,
        # )
        cross_attention = CrossAttentionModel(
            concept_dim, residual_dim, residual_dim, 8
        )
        return ConceptModel(
            base_network=make_cnn(bottleneck_dim, cnn_type=backbone),
            concept_network=Apply(lambda x: x[..., :concept_dim]),
            residual_network=Apply(lambda x: x[..., concept_dim:]),
            target_network=target_network,
            bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
            cross_attention=cross_attention,
            concept_rank_model=concept_rank_model,
            **config,
        )


def get_config(**kwargs) -> dict:
    config = {
        # ('model_type', 'beta'): ray.tune.grid_search([
        #     ('latent_residual', 0),
        #     ('decorrelated_residual', 10.0),
        #     ('iter_norm', 0),
        #     ('mi_residual', 1.0),
        # ]),
        "residual_dim": ray.tune.grid_search([0, 1, 2, 4, 8, 16, 32, 64]),
        "dataset": "celeba",
        # "data_dir": os.environ.get("CONCEPT_DATA_DIR", "./data"),
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        "training_mode": "independent",
        "num_epochs": 300,
        "lr": 3e-4,
        "batch_size": 64,
        "alpha": 1.0,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 1e-5,
        "cw_alignment_frequency": 20,
        "checkpoint_frequency": 5,
        "gpu_memory_per_worker": "11000 MiB",
        "strategy": ray.train.lightning.RayDDPStrategy(find_unused_parameters=True),
    }
    config.update(kwargs)
    config = process_grid_search_tuples(config)
    return config
