import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from torchvision.models.resnet import resnet18, ResNet18_Weights

from evaluation import (
    test_concept_residual_correlation,
    test_negative_interventions,
    test_random_concepts,
    test_random_residual,
)
from loader import get_data_loaders
from models import ConceptBottleneckModel, ConceptWhiteningModel
from train import train
from utils import make_ffn, concept_model_accuracy
from datetime import datetime
from pathlib import Path
from ray import tune
from ray import air


def main(args_dict):
    ### Data

    train_loader, test_loader, CONCEPT_DIM, NUM_CLASSES = get_data_loaders(
        'cifar100', batch_size=64, data_dir=args_dict["data_dir"])
    
    args_dict["concept_dim"] = CONCEPT_DIM
    args_dict["num_classes"] = NUM_CLASSES
    args_dict["dataset_name"] = 'cifar100'
    


    ### Models

    def make_resnet(output_dim):
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, output_dim)
        return resnet

    def make_bottleneck_model(residual_dim, config={}):
        return ConceptBottleneckModel(
            concept_network=nn.Sequential(make_resnet(CONCEPT_DIM), nn.Sigmoid()),
            residual_network=make_resnet(residual_dim),
            target_network=make_ffn(NUM_CLASSES),
            config = config
        )

    def make_whitening_model(residual_dim):
        bottleneck_dim = CONCEPT_DIM + residual_dim
        return ConceptWhiteningModel(
            base_network=make_resnet(bottleneck_dim),
            target_network=make_ffn(NUM_CLASSES),
            bottleneck_dim=bottleneck_dim,
        )
    
    dataset_name = train_loader.dataset.__class__.__name__
    save_dir = args_dict["save_dir"]
    date = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
    save_dir = Path(save_dir).resolve() / dataset_name / date
    save_dir.mkdir(parents=True, exist_ok=True)
    name = f"{str(dataset_name)}/{str(date)}"

    def trainable(config):
        train(
            config=config,
            make_bottleneck_model_fn=make_bottleneck_model,
            make_whitening_model_fn=make_whitening_model,
        )
    # do ray tune stuff here
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 1, "gpu": args_dict["num_gpus"]}),
        tune_config=tune.TuneConfig(
            num_samples=args_dict["num_samples"],
        ),
        param_space=args_dict,
        run_config=air.RunConfig(
            # name of your experiment
            # If the experiment with the same name is already run,
            # Tuner willl resume from the last run specified by sync_config(if one exists).
            # Otherwise, will start a new run.
            name=name,
            # a directory where results are stored before being
            # sync'd to head node/cloud storage
            local_dir=args_dict["ray_storage_dir"],
            # sync our checkpoints via rsync
            # you don't have to pass an empty sync config - but we
            # do it here for clarity and comparison
            checkpoint_config=air.CheckpointConfig(
                # we'll keep the best five checkpoints at all times
                # checkpoints (by AUC score, reported by the trainable, descending)
                checkpoint_score_attribute="test_acc",
                checkpoint_score_order="max",
                num_to_keep=5,
            ),
            # stopping criteria -- either timestep or max score reached
            # stop={
            #     "agent_timesteps_total": args.max_ts,
            #     "episode_reward_mean": args.max_score,
            # },
        ),
    )
    tuner.fit()

    
