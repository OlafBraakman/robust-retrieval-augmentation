import os
import random
import argparse
import torch
from torch.utils.data import Subset
from data.datasets.imagenet.pytorch_dataset import ImageNet
from data.datasets.cifar100.pytorch_dataset import CIFAR100
from tqdm import tqdm

def sample_balanced_subset(dataset, samples_per_class=2, total_classes=100, seed=42):
    random.seed(seed)

    # Dictionary of label -> list of indices
    label_to_indices = {label: [] for label in range(total_classes)}

    print("Indexing dataset by label...")
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in tqdm(indices):
        label = dataset.load_label(idx)
        if len(label_to_indices[label]) < samples_per_class:
            label_to_indices[label].append(idx)

    # Ensure each class has enough examples
    for label, indices in label_to_indices.items():
        if len(indices) != samples_per_class:
            raise ValueError(f"Label {label} has only {len(indices)} samples, need at least {samples_per_class}")

    print("Sampling subset indices...")
    sampled_indices = []
    for label in tqdm(range(total_classes)):
        sampled = random.sample(label_to_indices[label], samples_per_class)
        sampled_indices.extend(sampled)

    print(f"Sampled {len(sampled_indices)} total indices.")
    return sampled_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample 100k balanced ImageNet64 images")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train or val")
    parser.add_argument("--save-indices", type=str, default=None, help="Path to save sampled indices")
    args = parser.parse_args()

    dataset = CIFAR100(data_dir=args.data_dir, split=args.split)
    sampled_indices = sample_balanced_subset(dataset)

    subset = Subset(dataset, sampled_indices)

    if args.save_indices:
        torch.save(sampled_indices, args.save_indices)
        print(f"Saved sampled indices to {args.save_indices}")