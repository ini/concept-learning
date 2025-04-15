from __future__ import annotations

import os
import math
import argparse
import torch
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


def make_save_dir(save_name):
    """Create directory for saving embeddings if it doesn't exist."""
    save_dir = os.path.dirname(save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def save_clip_image_features(model, dataset, save_path, batch_size=1000, device="cuda"):
    """Process and save CLIP image embeddings as a single tensor file."""
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Check if file already exists
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping processing.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,  # Important: keep order consistent with indices
    )

    # Pre-allocate tensor for all features
    all_features = None

    with torch.no_grad():
        idx_offset = 0
        for (images, _), _ in tqdm(dataloader, desc="Processing images"):
            features = model.encode_image(images.to(device))
            features_cpu = features.cpu()

            # Store batch in the pre-allocated tensor
            batch_size_actual = features_cpu.shape[0]

            if all_features is None:
                all_features = torch.zeros((len(dataset), features_cpu.shape[1]))

            all_features[idx_offset : idx_offset + batch_size_actual] = features_cpu

            idx_offset += batch_size_actual

    # Save the entire tensor to a single file
    torch.save(all_features, save_path)
    print(f"Saved all {len(dataset)} image embeddings to {save_path}")
    torch.cuda.empty_cache()


def save_clip_text_features(model, text, save_path, batch_size=1000, device="cuda"):
    """Process and save CLIP text embeddings as a single tensor file."""
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Check if file already exists
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping processing.")
        return

    # Prepare text inputs for CLIP
    text_inputs = [f"a photo of a {concept}" for concept in text]
    text_tokens = clip.tokenize(text_inputs).to(device)

    with torch.no_grad():
        # Process a single token to get feature dimension
        sample_features = model.encode_text(text_tokens[0:1])
        feature_dim = sample_features.shape[1]

    # Pre-allocate tensor for all text features
    all_features = torch.zeros((len(text), feature_dim))

    with torch.no_grad():
        for i in tqdm(
            range(math.ceil(len(text_tokens) / batch_size)),
            desc="Processing text concepts",
        ):
            start_idx = batch_size * i
            end_idx = min(batch_size * (i + 1), len(text_tokens))

            batch_tokens = text_tokens[start_idx:end_idx]
            features = model.encode_text(batch_tokens)
            features_cpu = features.cpu()

            # Store batch in the pre-allocated tensor
            all_features[start_idx:end_idx] = features_cpu

    # Save all concept embeddings to a single file
    torch.save(all_features, save_path)

    # Also save the concept names for reference
    concepts_path = save_path.replace(".pt", "_names.txt")
    with open(concepts_path, "w") as f:
        for idx, concept in enumerate(text):
            f.write(f"{idx}: {concept}\n")

    print(f"Saved all {len(text)} concept embeddings to {save_path}")
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for datasets"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="Dataset to use (currently only cifar100 supported)",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="Root directory for the dataset"
    )

    # CLIP parameters
    parser.add_argument(
        "--clip_model", type=str, default="ViT-B/16", help="CLIP model to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )

    # Concept parameters
    parser.add_argument(
        "--concept_set_path",
        type=str,
        default=None,
        help="Path to the concept set file",
    )

    args = parser.parse_args()
    args.output_dir = os.path.join(args.data_root, args.clip_model)

    # Create base output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load CLIP model
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, clip_preprocess = clip.load(args.clip_model, device=args.device)
    print("CLIP model loaded successfully")

    if args.dataset.lower() == "cifar100":
        from datasets.cifar import CIFAR100, CLASSES, SUPERCLASSES

        print("Processing CIFAR-100 dataset")

        # Initialize datasets
        train_dataset = CIFAR100(
            root=args.data_root, train=True, transform=clip_preprocess, download=True
        )

        test_dataset = CIFAR100(
            root=args.data_root, train=False, transform=clip_preprocess, download=True
        )

        # Create model-specific directories
        model_name = args.clip_model.replace("/", "_")

        # Save paths for features
        train_save_path = os.path.join(args.output_dir, f"train.pt")
        test_save_path = os.path.join(args.output_dir, f"test.pt")

        # Process and save image embeddings
        print("Processing training set")
        save_clip_image_features(
            model=clip_model,
            dataset=train_dataset,
            save_path=train_save_path,
            batch_size=args.batch_size,
            device=args.device,
        )

        print("Processing test set")
        save_clip_image_features(
            model=clip_model,
            dataset=test_dataset,
            save_path=test_save_path,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Save labels mapping for reference
        with open(os.path.join(args.output_dir, "cifar100_classes.txt"), "w") as f:
            for idx, class_name in enumerate(CLASSES):
                f.write(f"{idx}: {class_name}\n")

        with open(os.path.join(args.output_dir, "cifar100_superclasses.txt"), "w") as f:
            for idx, superclass_name in enumerate(sorted(SUPERCLASSES.keys())):
                f.write(f"{idx}: {superclass_name}\n")

        # Process concepts if provided
        if args.concept_set_path:
            print(f"Processing concepts from {args.concept_set_path}")
            with open(args.concept_set_path) as f:
                concepts = f.read().strip().split("\n")

            concept_filename = os.path.basename(args.concept_set_path).split(".")[0]
            concepts_save_path = os.path.join(
                args.output_dir,
                f"concepts_{concept_filename}_{model_name}.pt",
            )

            save_clip_text_features(
                model=clip_model,
                text=concepts,
                save_path=concepts_save_path,
                batch_size=args.batch_size,
                device=args.device,
            )
        else:
            # If no concept file provided, use the CIFAR class names
            print("No concept set provided. Using CIFAR-100 class names as concepts.")
            classes_save_path = os.path.join(
                args.output_dir, f"concepts_cifar100_classes_{model_name}.pt"
            )

            save_clip_text_features(
                model=clip_model,
                text=CLASSES,
                save_path=classes_save_path,
                batch_size=args.batch_size,
                device=args.device,
            )

            # Also save superclass concepts
            superclass_concepts = sorted(SUPERCLASSES.keys())
            superclasses_save_path = os.path.join(
                args.output_dir, f"concepts_cifar100_superclasses_{model_name}.pt"
            )

            save_clip_text_features(
                model=clip_model,
                text=superclass_concepts,
                save_path=superclasses_save_path,
                batch_size=args.batch_size,
                device=args.device,
            )
    else:
        print(f"Dataset {args.dataset} not supported yet")

    print("Processing complete!")


if __name__ == "__main__":
    main()
