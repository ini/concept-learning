from __future__ import annotations
import os
import copy
import json
from tqdm import tqdm
import random
from tqdm import tqdm

# make sure slurm isn't exposed
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]

if "SLURM_JOB_NAME" in os.environ:
    del os.environ["SLURM_JOB_NAME"]
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
import argparse
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import ray.train

from copy import deepcopy
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray import air, tune
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import DATASET_INFO, get_concept_loss_fn
from lightning_ray import LightningTuner
from nn_extensions import Chain
from models import ConceptLightningModel
from models.mutual_info import MutualInformationLoss
from models.posthoc_concept_pred import (
    ConceptResidualConceptPred,
    ConceptEmbeddingConceptPred,
)
from train import make_concept_model, make_datamodule
from utils import cross_correlation, set_cuda_visible_devices


import os
import math
import torch

# import data_utils

from tqdm import tqdm
from torch.utils.data import DataLoader
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset

PM_SUFFIX = {"max": "_max", "avg": ""}


def get_similarity_from_activations(
    target_save_name,
    clip_save_name,
    text_save_name,
    similarity_fn,
    return_target_feats=True,
):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = image_features @ text_features.T
    del image_features, text_features
    torch.cuda.empty_cache()

    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)

    del clip_feats
    torch.cuda.empty_cache()

    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity


def get_activation(outputs, mode):
    """
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    """
    if mode == "avg":

        def hook(model, input, output):
            if len(output.shape) == 4:
                outputs.append(output.mean(dim=[2, 3]).detach().cpu())
            elif len(output.shape) == 2:
                outputs.append(output.detach().cpu())

    elif mode == "max":

        def hook(model, input, output):
            if len(output.shape) == 4:
                outputs.append(output.amax(dim=[2, 3]).detach().cpu())
            elif len(output.shape) == 2:
                outputs.append(output.detach().cpu())

    return hook


def get_save_names(
    clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir
):

    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(
            save_dir, d_probe, target_name.replace("/", "")
        )
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(
            save_dir, d_probe, target_name, target_layer, PM_SUFFIX[pool_mode]
        )
    clip_save_name = "{}/{}_clip_{}.pt".format(
        save_dir, d_probe, clip_name.replace("/", "")
    )
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(
        save_dir, concept_set_name, clip_name.replace("/", "")
    )

    return target_save_name, clip_save_name, text_save_name


def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True


def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[: save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return


def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    for images, labels in tqdm(
        DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
    ):
        with torch.no_grad():
            # outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu() == labels)
            total += len(labels)
    return correct / total


def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(
        DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
    ):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def get_save_path(
    plot_key: tuple,
    prefix: str | None = None,
    suffix: str | None = None,
    save_dir: Path | str = "./plots",
) -> Path:
    """
    Get the save path for the given plot.
    """
    items = [str(key).replace(".", "_") for key in plot_key]
    if prefix:
        items.insert(0, prefix)
    if suffix:
        items.append(suffix)

    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    return save_dir / "_".join(items)


def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(
        DataLoader(dataset, 500, num_workers=8, pin_memory=True)
    ):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred = []
    for i in range(torch.max(pred) + 1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds == i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred


import torch
import math
from tqdm import tqdm


def cos_similarity_cubed_single(clip_feats, target_feats):
    """
    Substract mean from each vector, then raises to third power and compares cos similarity
    Does not modify any tensors in place
    Only compares first neuron to first concept etc.
    """

    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)

    clip_feats = clip_feats**3
    target_feats = target_feats**3

    clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=0, keepdim=True)
    target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)

    similarities = torch.sum(target_feats * clip_feats, dim=0)
    return similarities


def train_posthoc_fitter(
    model: ConceptLightningModel,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_train_epochs: int = 10,
    dataset=None,
    data_dir=None,
    num_concepts=None,
    backbone=None,
    subset=None,
    config={},
    use_concept_res=True,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_train_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    if dataset == "cifar100":
        num_classes = 100
    elif dataset == "cub":
        num_classes = 200
    clip_path = os.path.join(config["data_dir"], "ViT-B/16/")

    concept_loss_fn = get_concept_loss_fn(
        dataset,
        data_dir,
        num_concepts=num_concepts,
        backbone=backbone,
        subset=subset,
    )

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with open(config["concept_set_path"]) as f:
        concepts = f.read().split("\n")
    if not os.path.exists(config["save_dir"] / "final_features"):
        os.makedirs(config["save_dir"] / "final_features", exist_ok=True)
        for ds, split in zip(
            [train_loader, val_loader, test_loader],
            ["train", "val", "test"],
        ):
            save_residual_preds(
                model, ds, clip_path, config["save_dir"] / "final_features", split
            )

    # Load saved tensors
    (
        train_residuals,
        train_concept_preds,
        train_concepts,
        train_targets,
        train_clip_features,
    ) = load_featurees(config, "train", clip_path)
    concept_clip_features = torch.load(
        os.path.join(clip_path, "concepts_cifar100_filtered_ViT-B_16.pt")
    )
    # load features
    with torch.no_grad():
        image_features = train_clip_features
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = concept_clip_features
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T

        del image_features, text_features

    # filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

    if True:
        for i, concept in enumerate(concepts):
            if highest[i] <= config["clip_cutoff"]:
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
    concepts = [
        concepts[i] for i in range(len(concepts)) if highest[i] > config["clip_cutoff"]
    ]

    with torch.no_grad():
        image_features = train_clip_features
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = concept_clip_features[highest > config["clip_cutoff"]]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T

        del image_features, text_features

    (
        val_residuals,
        val_concept_preds,
        val_concepts,
        val_targets,
        val_clip_features,
    ) = load_featurees(config, "val", clip_path)

    with torch.no_grad():
        image_features = val_clip_features
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = concept_clip_features[highest > config["clip_cutoff"]]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        val_clip_features = image_features @ text_features.T

        del image_features, text_features

    # learn projection layer
    proj_layer_size = (
        train_residuals.shape[1]
        if not use_concept_res
        else train_residuals.shape[1] + train_concepts.shape[1]
    )
    proj_layer = torch.nn.Linear(
        in_features=proj_layer_size, out_features=len(concepts), bias=False
    ).to(model.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-4)

    indices = [ind for ind in range(len(train_residuals))]

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(config["proj_batch_size"], len(train_residuals))
    similarity_fn = cos_similarity_cubed_single
    if use_concept_res:
        weight_name = f"best_proj_weights_cr{proj_batch_size}.pt"
    else:
        weight_name = "best_proj_weights.pt"
    if (
        True
        or not os.path.exists(os.path.join(config["save_dir"], weight_name))
        or config.get("overwrite_proj_weights", False)
    ):
        for i in range(config["proj_steps"]):
            batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
            if use_concept_res:
                train_residuals_batch = torch.cat(
                    [train_residuals[batch], train_concepts[batch]], dim=1
                )
                outs = proj_layer(train_residuals_batch.to(model.device).detach())
            else:
                outs = proj_layer(train_residuals[batch].to(model.device).detach())
            loss = -similarity_fn(clip_features[batch].to(model.device).detach(), outs)

            loss = torch.mean(loss)
            loss.backward()
            opt.step()
            if i % 50 == 0 or i == config["proj_steps"] - 1:
                with torch.no_grad():
                    if use_concept_res:
                        val_residuals_batch = torch.cat(
                            [val_residuals, val_concepts], dim=1
                        )
                        val_output = proj_layer(
                            val_residuals_batch.to(model.device).detach()
                        )
                    else:
                        val_output = proj_layer(val_residuals.to(model.device).detach())
                    val_loss = -similarity_fn(
                        val_clip_features.to(model.device).detach(), val_output
                    )
                    total_concepts = val_loss.shape[0]
                    # Count concepts with similarity above 0.45
                    high_similarity_count = torch.sum((-val_loss) > 0.45).item()
                    val_loss = torch.mean(val_loss)

                if i == 0:
                    best_val_loss = val_loss
                    best_step = i
                    best_weights = proj_layer.weight.clone()
                    print(
                        "Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}, Concepts with sim > 0.45: {}/{}".format(
                            best_step,
                            -loss.cpu(),
                            -best_val_loss.cpu(),
                            high_similarity_count,
                            total_concepts,
                        )
                    )

                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = i
                    best_weights = proj_layer.weight.clone()
                    print(
                        "Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}, Concepts with sim > 0.45: {}/{}".format(
                            best_step,
                            -loss.cpu(),
                            -best_val_loss.cpu(),
                            high_similarity_count,
                            total_concepts,
                        )
                    )
                else:  # stop if val loss starts increasing
                    break
            opt.zero_grad()

        torch.save(
            best_weights.cpu(),
            os.path.join(config["save_dir"], weight_name),
        )
        print(
            "Best step:{}, Avg val similarity:{:.4f}".format(
                best_step, -best_val_loss.cpu()
            )
        )
    else:
        best_weights = torch.load(os.path.join(config["save_dir"], weight_name)).to(
            model.device
        )

    proj_layer.load_state_dict({"weight": best_weights})

    # delete concepts that are not interpretable
    with torch.no_grad():
        if use_concept_res:
            val_residuals_batch = torch.cat([val_residuals, val_concepts], dim=1)
            outs = proj_layer(val_residuals_batch.to(model.device).detach())
        else:
            outs = proj_layer(val_residuals.to(model.device).detach())
        sim = similarity_fn(val_clip_features.to(model.device).detach(), outs)
        interpretable = sim > config["interpretability_cutoff"]
        if interpretable.sum() > 256:
            top_indices = sim.argsort(descending=True)[:256]
            interpretable = torch.zeros_like(sim, dtype=torch.bool)
            interpretable[top_indices] = True

    # if True:
    #     for i, concept in enumerate(concepts):
    #         if sim[i] <= config["interpretability_cutoff"]:
    #             print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
    # assert 0, interpretable.sum().item()

    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]

    del clip_features, val_clip_features

    W_c = proj_layer.weight[interpretable]

    proj_layer = torch.nn.Linear(
        in_features=proj_layer_size, out_features=len(concepts), bias=False
    )
    proj_layer.load_state_dict({"weight": W_c})

    with torch.no_grad():
        if use_concept_res:
            train_residuals_batch = torch.cat(
                [train_residuals, train_concept_preds], dim=1
            )
            train_r = proj_layer(train_residuals_batch.detach())

            val_residuals_batch = torch.cat([val_residuals, val_concept_preds], dim=1)
            val_r = proj_layer(val_residuals_batch.detach())
        else:
            train_r = proj_layer(train_residuals.detach())
            val_r = proj_layer(val_residuals.detach())
        # train_cr = train_r
        # val_cr = val_r
        train_cr = torch.cat([train_r, train_concept_preds], dim=1)
        val_cr = torch.cat([val_r, val_concept_preds], dim=1)

        train_mean = torch.mean(train_cr, dim=0, keepdim=True)
        train_std = torch.std(train_cr, dim=0, keepdim=True)
        if config.get("no_glm", False):
            train_mean = torch.zeros_like(train_mean)
            train_std = torch.ones_like(train_std)

        train_cr -= train_mean
        train_cr /= train_std

        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(train_cr, train_y)

        val_cr -= train_mean
        val_cr /= train_std

        val_y = torch.LongTensor(val_targets)

        val_ds = TensorDataset(val_cr, val_y)

    indexed_train_loader = DataLoader(
        indexed_train_ds, batch_size=config["saga_batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=config["saga_batch_size"], shuffle=False)

    if config.get("no_glm", False):
        # Use Adam optimizer instead of GLM SAGA
        linear = torch.nn.Linear(train_cr.shape[1], num_classes).to(model.device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(
            linear.parameters(), lr=config.get("adam_lr", 1e-3)
        )

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        num_epochs = 30
        best_val_acc = 0
        best_state = None

        for epoch in range(num_epochs):
            # Training
            linear.train()
            train_loss = 0.0

            for out in indexed_train_loader:
                inputs, targets, _ = out
                inputs, targets = inputs.to(model.device), targets.to(model.device)

                # Forward pass
                outputs = linear(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(indexed_train_ds)

            # Validation
            linear.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(model.device), targets.to(model.device)

                    outputs = linear(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            val_loss /= len(val_ds)
            val_acc = correct / total

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "weight": linear.weight.data.clone(),
                    "bias": linear.bias.data.clone(),
                }

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

        # Use the best model weights
        if best_state is not None:
            W_g = best_state["weight"]
            b_g = best_state["bias"]
        else:
            W_g = linear.weight.data
            b_g = linear.bias.data

    else:

        # Make linear model and zero initialize
        linear = torch.nn.Linear(train_cr.shape[1], num_classes).to(model.device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()

        STEP_SIZE = 0.1
        ALPHA = 0.99
        metadata = {}
        metadata["max_reg"] = {}
        metadata["max_reg"]["nongrouped"] = config["lam"]

        # Solve the GLM path
        output_proj = glm_saga(
            linear,
            indexed_train_loader,
            STEP_SIZE,
            config["n_iters"],
            ALPHA,
            epsilon=1,
            k=1,
            val_loader=val_loader,
            do_zero=False,
            metadata=metadata,
            n_ex=len(train_cr),
            n_classes=num_classes,
            verbose=10,
        )
        W_g = output_proj["path"][0]["weight"]
        b_g = output_proj["path"][0]["bias"]

        import datetime

        torch.save(train_mean, os.path.join(config["save_dir"], "proj_mean.pt"))
        torch.save(train_std, os.path.join(config["save_dir"], "proj_std.pt"))
        torch.save(W_c, os.path.join(config["save_dir"], "W_c.pt"))
        torch.save(W_g, os.path.join(config["save_dir"], "W_g.pt"))
        torch.save(b_g, os.path.join(config["save_dir"], "b_g.pt"))

        if len(concepts) > 0:
            with open(
                os.path.join(config["save_dir"], "chosen_concepts.txt"), "w"
            ) as f:
                f.write(concepts[0])
                for concept in concepts[1:]:
                    f.write("\n" + concept)

        with open(os.path.join(config["save_dir"], "metrics.txt"), "w") as f:
            out_dict = {}
            for key in ("lam", "lr", "alpha", "time"):
                out_dict[key] = float(output_proj["path"][0][key])
            out_dict["metrics"] = output_proj["path"][0]["metrics"]
            nnz = (W_g.abs() > 1e-5).sum().item()
            total = W_g.numel()
            out_dict["sparsity"] = {
                "Non-zero weights": nnz,
                "Total weights": total,
                "Percentage non-zero": nnz / total,
            }
            json.dump(out_dict, f, indent=2)
    (
        test_residuals,
        test_concept_preds,
        test_concepts,
        test_targets,
        _,
    ) = load_featurees(config, "test", clip_path)

    final = torch.nn.Linear(in_features=W_g.shape[1], out_features=W_g.shape[0]).to(
        model.device
    )
    final.load_state_dict({"weight": W_g, "bias": b_g})

    # evaluate performance
    if use_concept_res:
        test_residuals_batch = torch.cat([test_residuals, test_concept_preds], dim=1)
        test_r = proj_layer(test_residuals_batch.detach())

        test_residuals_batch_int = torch.cat([test_residuals, test_concepts], dim=1)
        test_r_int = proj_layer(test_residuals_batch_int.detach())

    else:
        test_r = proj_layer(test_residuals.detach())
        test_r_int = test_r
    test_cr = torch.cat([test_r, test_concept_preds], dim=1)
    test_cr_int = torch.cat([test_r_int, test_concepts], dim=1)
    # test_cr = test_r
    # test_cr_int = test_r_int

    test_cr -= train_mean
    test_cr /= train_std

    test_cr_int -= train_mean
    test_cr_int /= train_std
    test_cr = test_cr.to(model.device)
    test_cr_int = test_cr_int.to(model.device)
    test_y_pred = final(test_cr).cpu()
    test_y_pred = torch.argmax(test_y_pred, dim=1)
    test_y_acc = torch.sum(test_y_pred == test_targets).item() / len(test_targets)
    test_y_pred_int = final(test_cr_int).cpu()
    test_y_pred_int = torch.argmax(test_y_pred_int, dim=1)
    test_y_acc_int = torch.sum(test_y_pred_int == test_targets).item() / len(
        test_targets
    )
    print(
        "Test accuracy: {:.4f}, Test accuracy with intervention: {:.4f}".format(
            test_y_acc, test_y_acc_int
        )
    )
    return interpretable.sum().item(), test_y_acc, test_y_acc_int


def train_posthoc_crm(
    model: ConceptLightningModel,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_train_epochs: int = 10,
    dataset=None,
    data_dir=None,
    num_concepts=None,
    backbone=None,
    subset=None,
    config={},
    use_concept_res=True,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_train_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    if dataset == "cifar100":
        num_classes = 100
    elif dataset == "cub":
        num_classes = 200
    clip_path = os.path.join(config["data_dir"], "ViT-B/16/")

    concept_loss_fn = get_concept_loss_fn(
        dataset,
        data_dir,
        num_concepts=num_concepts,
        backbone=backbone,
        subset=subset,
    )

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with open(config["concept_set_path"]) as f:
        concepts = f.read().split("\n")
    if not os.path.exists(config["save_dir"] / "final_features"):
        os.makedirs(config["save_dir"] / "final_features", exist_ok=True)
        for ds, split in zip(
            [train_loader, val_loader, test_loader],
            ["train", "val", "test"],
        ):
            save_residual_preds(
                model, ds, clip_path, config["save_dir"] / "final_features", split
            )
    (
        train_residuals,
        train_concept_preds,
        train_concepts,
        train_targets,
        train_clip_features,
    ) = load_featurees(config, "train", clip_path)

    (
        val_residuals,
        val_concept_preds,
        val_concepts,
        val_targets,
        val_clip_features,
    ) = load_featurees(config, "val", clip_path)

    train_cr = torch.cat([train_residuals, train_concepts], dim=1)
    val_cr = torch.cat([val_residuals, val_concepts], dim=1)

    train_mean = torch.mean(train_cr, dim=0, keepdim=True)
    train_std = torch.std(train_cr, dim=0, keepdim=True)
    if config.get("no_glm", False):
        train_mean = torch.zeros_like(train_mean)
        train_std = torch.ones_like(train_std)

    train_cr -= train_mean
    train_cr /= train_std

    val_cr -= train_mean
    val_cr /= train_std

    train_y = torch.LongTensor(train_targets)
    indexed_train_ds = IndexedTensorDataset(train_cr, train_y)
    val_y = torch.LongTensor(val_targets)
    val_ds = TensorDataset(val_cr, val_y)

    indexed_train_loader = DataLoader(
        indexed_train_ds, batch_size=config["saga_batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=config["saga_batch_size"], shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_cr.shape[1], num_classes).to(model.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    # Check if config["no_glm"] is on
    if config.get("no_glm", False):
        # Use Adam optimizer instead of GLM SAGA
        linear = torch.nn.Linear(train_cr.shape[1], num_classes).to(model.device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(
            linear.parameters(), lr=config.get("adam_lr", 1e-3)
        )

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        num_epochs = 30
        best_val_acc = 0
        best_state = None

        for epoch in range(num_epochs):
            # Training
            linear.train()
            train_loss = 0.0

            for out in indexed_train_loader:
                inputs, targets, _ = out
                inputs, targets = inputs.to(model.device), targets.to(model.device)

                # Forward pass
                outputs = linear(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(indexed_train_ds)

            # Validation
            linear.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(model.device), targets.to(model.device)

                    outputs = linear(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            val_loss /= len(val_ds)
            val_acc = correct / total

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "weight": linear.weight.data.clone(),
                    "bias": linear.bias.data.clone(),
                }

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

        # Use the best model weights
        if best_state is not None:
            W_g = best_state["weight"]
            b_g = best_state["bias"]
        else:
            W_g = linear.weight.data
            b_g = linear.bias.data

    else:

        STEP_SIZE = 0.1
        ALPHA = 0.99
        metadata = {}
        metadata["max_reg"] = {}
        metadata["max_reg"]["nongrouped"] = config["lam"]

        # Solve the GLM path
        output_proj = glm_saga(
            linear,
            indexed_train_loader,
            STEP_SIZE,
            config["n_iters"],
            ALPHA,
            epsilon=1,
            k=1,
            val_loader=val_loader,
            do_zero=False,
            metadata=metadata,
            n_ex=len(train_cr),
            n_classes=num_classes,
            verbose=10,
        )
        W_g = output_proj["path"][0]["weight"]
        b_g = output_proj["path"][0]["bias"]

    (
        test_residuals,
        test_concept_preds,
        test_concepts,
        test_targets,
        _,
    ) = load_featurees(config, "test", clip_path)

    final = torch.nn.Linear(in_features=W_g.shape[1], out_features=W_g.shape[0]).to(
        model.device
    )
    final.load_state_dict({"weight": W_g, "bias": b_g})
    test_cr = torch.cat([test_residuals, test_concept_preds], dim=1)
    test_cr_int = torch.cat([test_residuals, test_concepts], dim=1)
    # test_cr = test_r
    # test_cr_int = test_r_int

    test_cr -= train_mean
    test_cr /= train_std

    test_cr_int -= train_mean
    test_cr_int /= train_std
    test_cr = test_cr.to(model.device)
    test_cr_int = test_cr_int.to(model.device)
    test_y_pred = final(test_cr).cpu()
    test_y_pred = torch.argmax(test_y_pred, dim=1)
    test_y_acc = torch.sum(test_y_pred == test_targets).item() / len(test_targets)
    test_y_pred_int = final(test_cr_int).cpu()
    test_y_pred_int = torch.argmax(test_y_pred_int, dim=1)
    test_y_acc_int = torch.sum(test_y_pred_int == test_targets).item() / len(
        test_targets
    )
    print(
        "Test accuracy: {:.4f}, Test accuracy with intervention: {:.4f}".format(
            test_y_acc, test_y_acc_int
        )
    )
    return 0, test_y_acc, test_y_acc_int


def load_featurees(config, split, clip_path):
    indices = torch.load(
        os.path.join(config["save_dir"], "final_features", f"{split}_indices.pt")
    )
    residuals = torch.load(
        os.path.join(config["save_dir"], "final_features", f"{split}_residuals.pt")
    )
    concept_preds = torch.load(
        os.path.join(config["save_dir"], "final_features", f"{split}_concept_preds.pt")
    )
    concepts = torch.load(
        os.path.join(config["save_dir"], "final_features", f"{split}_concepts.pt")
    )
    targets = torch.load(
        os.path.join(config["save_dir"], "final_features", f"{split}_targets.pt")
    )

    # Load CLIP features
    if split == "val" and not os.path.exists(os.path.join(clip_path, "val.pt")):
        clip_features_og = torch.load(os.path.join(clip_path, "train.pt"))
    else:
        clip_features_og = torch.load(os.path.join(clip_path, f"{split}.pt"))

    clip_features = clip_features_og[indices]
    return residuals, concept_preds, concepts, targets, clip_features


def save_residual_preds(
    model: ConceptLightningModel,
    ds_loader: DataLoader,
    clip_path: str,
    save_dir: str,
    split: str = "train",
) -> float:
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    # Get mutual information estimator
    # (data, concepts), targets = next(iter(test_loader))
    # data = data.to(model.device)
    # concepts = concepts.to(model.device)
    # _, residual, _ = model(data, concepts=concepts)
    # model.concept_model.training = False
    # concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
    # base_ds = test_loader.dataset
    from sklearn.metrics import confusion_matrix
    import os

    # Initialize lists to store true labels and predictions
    model.eval()

    # Initialize lists to accumulate tensors
    all_indices = []
    all_residuals = []
    all_concept_preds = []
    all_concepts = []
    all_targets = []

    # Iterate through the test loader
    for indicies, (data, concepts), target in ds_loader:
        data = data.to(model.device)
        concepts = concepts.to(model.device)
        target = target.to(model.device)

        with torch.no_grad():
            intervention_idxs = torch.ones_like(concepts)
            # intervene on the white concept
            concept_logits, residual, y_pred_polar = model(
                data,
                concepts=concepts,
                intervention_idxs=intervention_idxs,
            )
            concept_preds = model.concept_model.get_concept_predictions(concept_logits)
            if type(residual) == tuple:
                residual = residual[0]

            # Accumulate tensors
            all_indices.append(indicies.cpu())
            all_residuals.append(residual.cpu())
            all_concept_preds.append(concept_preds.cpu())
            all_concepts.append(concepts.cpu())
            all_targets.append(target.cpu())

    # Concatenate all accumulated tensors
    all_indices = torch.cat(all_indices, dim=0)
    all_residuals = torch.cat(all_residuals, dim=0)
    all_concept_preds = torch.cat(all_concept_preds, dim=0)
    all_concepts = torch.cat(all_concepts, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save tensors to files
    torch.save(all_indices, os.path.join(save_dir, f"{split}_indices.pt"))
    torch.save(all_residuals, os.path.join(save_dir, f"{split}_residuals.pt"))
    torch.save(all_concept_preds, os.path.join(save_dir, f"{split}_concept_preds.pt"))
    torch.save(all_concepts, os.path.join(save_dir, f"{split}_concepts.pt"))
    torch.save(all_targets, os.path.join(save_dir, f"{split}_targets.pt"))

    print(f"Saved all tensors to {save_dir}")


### Loading & Execution


def filter_eval_configs(configs: list[dict]) -> list[dict]:
    """
    Filter evaluation configs.

    Parameters
    ----------
    configs : list[dict]
        List of evaluation configs
    """
    configs_to_keep = []
    for config in configs:
        if config["model_type"] == "concept_whitening":
            if config["eval_mode"].endswith("intervention"):
                print("Interventions not supported for concept whitening models")
                continue

        if config["model_type"] == "no_residual" or config["residual_dim"] == 0:
            if config["eval_mode"] in ("correlation", "mutual_info"):
                print("Correlation / MI metrics not available for no-residual models")
                continue

        configs_to_keep.append(config)

    return configs_to_keep


def evaluate(config: dict):
    """
    Evaluate a trained model.

    Parameters
    ----------
    config : dict
        Evaluation configuration dictionary
    """
    metrics = {}
    # Get data loader
    if config["dataset"] == "celeba" or config["dataset"] == "pitfalls_synthetic":
        new_config = copy.deepcopy(config)
        new_config["num_concepts"] = 8
        new_config["batch_size"] = 256
        train_loader = make_datamodule(**new_config).train_dataloader()
        val_loader = make_datamodule(**new_config).val_dataloader()
        test_loader = make_datamodule(**new_config).test_dataloader()
    else:
        new_config = copy.deepcopy(config)

        train_loader = make_datamodule(**new_config).train_dataloader()
        val_loader = make_datamodule(**new_config).val_dataloader()
        test_loader = make_datamodule(**new_config).test_dataloader()

    from torch.utils.data import Dataset, DataLoader

    def get_original_index(dataset, idx):
        if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
            # This is a Subset, so trace back
            subset_idx = dataset.indices[idx]
            return get_original_index(dataset.dataset, subset_idx)
        else:
            # This is the original dataset, return the index
            return idx

    class IndexedDataset(Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            data = self.original_dataset[idx]

            # Return the index along with the data
            return (
                (get_original_index(self.original_dataset, idx), *data)
                if isinstance(data, tuple)
                else (get_original_index(self.original_dataset, idx), data)
            )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    # Wrap datasets to include indices
    indexed_train_dataset = IndexedDataset(train_dataset)
    indexed_val_dataset = IndexedDataset(val_dataset)
    indexed_test_dataset = IndexedDataset(test_dataset)

    # Create data loaders with indexed datasets
    train_loader = DataLoader(
        indexed_train_dataset, batch_size=new_config["batch_size"], shuffle=False
    )
    val_loader = DataLoader(
        indexed_val_dataset, batch_size=new_config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        indexed_test_dataset, batch_size=new_config["batch_size"], shuffle=False
    )

    # Load model
    tuner = LightningTuner("val_acc", "max")
    model = tuner.load_model(make_concept_model, config["train_result"])
    if config["dataset"] == "mimic_cxr":
        dataset_info = DATASET_INFO[config["dataset"]][config["subset"]]
    else:
        dataset_info = DATASET_INFO[config["dataset"]]

    # if config["dataset"] == "cifar100":
    #     ds_name = config["dataset"]
    #     config["concept_set_path"] = f"./posthoc_concept_sets/{ds_name}_filtered.txt"
    # else:
    #     assert 0, "Dataset not supported"
    cr = "_cr" if config.get("use_concept_res", True) else ""

    if config.get("no_cbm", False):
        metrics[f"posthoc_res"] = train_posthoc_crm(
            model,
            config["model_type"],
            train_loader,
            val_loader,
            test_loader,
            dataset=config["dataset"],
            data_dir=config["data_dir"],
            num_concepts=config.get("num_concepts", -1),
            backbone=config.get("backbone", "resnet34"),
            subset=config.get("subset", None),
            config=config,
            use_concept_res=config.get("use_concept_res", True),
        )
    else:
        metrics[f"posthoc_fitter{cr}"] = train_posthoc_fitter(
            model,
            config["model_type"],
            train_loader,
            val_loader,
            test_loader,
            dataset=config["dataset"],
            data_dir=config["data_dir"],
            num_concepts=config.get("num_concepts", -1),
            backbone=config.get("backbone", "resnet34"),
            subset=config.get("subset", None),
            config=config,
            use_concept_res=config.get("use_concept_res", True),
        )
    # Report evaluation metrics
    ray.train.report(metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        help="Experiment directory",
    )
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["dataset", "model_type"],
        help="Config keys to group by when selecting best trial results",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all trained models (instead of best trial per group)",
    )
    parser.add_argument(
        "--num-cpus", type=float, default=1, help="Number of CPUs to use (per model)"
    )
    parser.add_argument(
        "--num-gpus", type=float, default=1, help="Number of GPUs to use (per model)"
    )
    parser.add_argument(
        "--concept-set-path",
        type=str,
        default="/home/renos/concept-learning/posthoc_concept_sets/cifar100_filtered.txt",
    )
    parser.add_argument(
        "--clip-cutoff",
        type=float,
        default=0.25,
        help="Cutoff for CLIP activations",
    )
    parser.add_argument(
        "--saga_batch_size",
        type=int,
        default=256,
        help="Batch size used when fitting final layer",
    )
    parser.add_argument(
        "--proj_batch_size",
        type=int,
        default=50000,
        help="Batch size to use when learning projection layer",
    )

    parser.add_argument(
        "--clip_cutoff",
        type=float,
        default=0.25,
        help="concepts with smaller top5 clip activation will be deleted",
    )
    parser.add_argument(
        "--proj_steps",
        type=int,
        default=3000,
        help="how many steps to train the projection layer for",
    )
    parser.add_argument(
        "--interpretability_cutoff",
        type=float,
        default=0.45,
        help="concepts with smaller similarity to target concept will be deleted",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.00007,
        help="Sparsity regularization parameter, higher->more sparse",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=100,
        help="How many iterations to run the final layer solver for",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print all concepts being deleted in this stage",
    )
    parser.add_argument(
        "--use_concept_res",
        action="store_true",
        help="Use concept residuals when training the projection layer",
    )
    parser.add_argument(
        "--no_cbm",
        action="store_true",
        help="Use concept residuals when training the target network",
    )
    parser.add_argument(
        "--no_glm",
        action="store_true",
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob("**/train/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load train results
    train_folder = "train"
    print("Loading training results from", experiment_path / train_folder)
    tuner = LightningTuner.restore(experiment_path / train_folder)
    if args.all:
        results = tuner.get_results()
    else:
        results = [
            group.get_best_result()
            for group in tuner.get_results(groupby=args.groupby).values()
        ]

    # Create evaluation configs
    results = [result for result in results if result.config is not None]

    # Define default values for configuration
    default_config = {
        "clip_name": "clip_vitb32",
        "backbone": "resnet34",
        "feature_layer": "layer4",
        "concept_set": "cifar100",
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "activation_dir": "./activations",
        "pool_mode": "avg",
    }

    # Add default values to the configs
    eval_configs = []
    for result in results:
        config_ = {
            **default_config,  # Add all default values
            **result.config[
                "train_loop_config"
            ],  # Override with existing config values
            "train_result": result,
        }
        config_["save_dir"] = (
            experiment_path / "train_posthoccbm_resources" / result.path.split("/")[-1]
        )

        for key in args.__dict__:
            config_[key] = args.__dict__[key]
        eval_configs.append(config_)

    eval_configs = filter_eval_configs(eval_configs)

    # Get available resources
    if args.num_gpus < 1:
        set_cuda_visible_devices(available_memory_threshold=args.num_gpus)

    # Run evaluations
    eval_folder = "eval"
    tuner = tune.Tuner(
        tune.with_resources(
            evaluate,
            resources={
                "cpu": args.num_cpus,
                "gpu": args.num_gpus if torch.cuda.is_available() else 0,
            },
        ),
        param_space=tune.grid_search(eval_configs),
        run_config=air.RunConfig(name=eval_folder, storage_path=experiment_path),
    )
    eval_results = tuner.fit()
