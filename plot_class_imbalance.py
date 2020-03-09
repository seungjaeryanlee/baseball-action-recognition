"""
Plot histograms that show class imbalance.
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import video_transforms

from i3d import InceptionI3d
import bbdb_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == '__main__':
    CONFIG = {
        ## Data
        "DATASET": "original", # ["original", "binary"]
        "SEGMENT_LENGTH": 37,
        "FRAMESKIP": 4,

        ## Training
        # NOTE(seungjaeryanlee): Originally 8*5, but lowered due to memory
        "BATCH_SIZE": 2,
    }

    # Setup Datasets and Dataloaders
    if CONFIG["DATASET"] == "original":
        Dataset = bbdb_dataset.OriginalBBDBDataset
    elif CONFIG["DATASET"] == "binary":
        Dataset = bbdb_dataset.BinaryBBDBDataset
    elif CONFIG["DATASET"] == "debug":
        Dataset = bbdb_dataset.DebugBBDBDataset
    else:
        assert False

    # Setup Dataset and Dataloader
    with open("data_split.min.json", "r") as fp:
        data_split = json.load(fp)

    train_transforms = transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.RandomCrop(224),
        video_transforms.RandomHorizontalFlip(),
    ])
    val_transforms = transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.CenterCrop(224),
    ])
    test_transforms = transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.CenterCrop(224),
    ])
    train_dataset = Dataset(
        segment_filepaths=data_split["train"],
        segment_length=CONFIG["SEGMENT_LENGTH"],
        frameskip=CONFIG["FRAMESKIP"],
        transform=train_transforms,
    )
    dataloader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, pin_memory=True)

    val_dataset = Dataset(
        segment_filepaths=data_split["valid"],
        segment_length=CONFIG["SEGMENT_LENGTH"],
        frameskip=CONFIG["FRAMESKIP"],
        transform=val_transforms,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, pin_memory=True)
    
    test_dataset = Dataset(
        segment_filepaths=data_split["test"],
        segment_length=CONFIG["SEGMENT_LENGTH"],
        frameskip=CONFIG["FRAMESKIP"],
        transform=test_transforms,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], pin_memory=True)

    # Plot histogram of labels in datasets
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
    axs[0].set_title("Training set")
    axs[0].bar(range(train_dataset.NUM_LABELS), train_dataset.label_counts / sum(train_dataset.label_counts))
    axs[1].set_title("Validation set")
    axs[1].bar(range(val_dataset.NUM_LABELS), val_dataset.label_counts / sum(val_dataset.label_counts))
    axs[2].set_title("Test set")
    axs[2].bar(range(test_dataset.NUM_LABELS), test_dataset.label_counts / sum(test_dataset.label_counts))
    plt.show()
