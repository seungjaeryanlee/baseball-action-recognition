"""
View some frames in a video segment.
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import video_transforms
import matplotlib.pyplot as plt

from i3d import InceptionI3d
import bbdb_dataset
from tqdm import tqdm


if __name__ == '__main__':
    CONFIG = {
        ## Data
        "DATASET": "original",
        "SEGMENT_LENGTH": 150,
        "FRAMESKIP": 1,

        ## Training
        # NOTE(seungjaeryanlee): Originally 8*5, but lowered due to memory
        "BATCH_SIZE": 4,
    }
    # Setup Dataset and Dataloader
    if CONFIG["DATASET"] == "original":
        Dataset = bbdb_dataset.OriginalBBDBDataset
    elif CONFIG["DATASET"] == "binary":
        Dataset = bbdb_dataset.BinaryBBDBDataset
    else:
        assert False

    with open("data_split.min.json", "r") as fp:
        data_split = json.load(fp)
    test_transforms = transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.CenterCrop(224),
    ])
    dataset = Dataset(
        segment_filepaths=data_split["test"],
        segment_length=CONFIG["SEGMENT_LENGTH"],
        frameskip=CONFIG["FRAMESKIP"],
        transform=test_transforms
    )

    # Get video with wanted label
    i = 50
    video, label = dataset[i]
    label_index = (label[:, 0] == 1).nonzero()[0][0]
    i += 1
    while label_index != bbdb_dataset.LABEL_STR_TO_ID["Strike"]:
        video, label = dataset[i]
        label_index = (label[:, 0] == 1).nonzero()[0][0]
        i += 1
    print("Video segment: ", i)

    fig, axs = plt.subplots(6, 1, figsize=(4, 12))
    for i, image_index in enumerate([0, 29, 59, 89, 119, 149]):
        image = video[:, image_index, :, :].transpose(1, 2, 0)
        image = (image + 1) / 2 # Un-normalize
        axs[i].imshow(image)
    fig.tight_layout()
    plt.show()
