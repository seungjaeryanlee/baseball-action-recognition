"""
Define the PyTorch Dataset class for Baseball Database (BBDB).
"""
import glob
import json
import re

import numpy as np
import skvideo.io
from torch.utils.data import Dataset


NUM_LABELS = 30
LABEL_ID_TO_STR = {
    0: "Ball",
    1: "Strike",
    2: "Foul",
    3: "Swing and a miss",
    4: "Fly out",
    5: "Ground out",
    6: "One-base hit",
    7: "Strike out",
    8: "Home in",
    9: "Base on balls",
    10: "Touch out",
    11: "Two-base hit",
    12: "Homerun",
    13: "Foul fly out",
    14: "Double play",
    15: "Tag out",
    16: "Stealing base",
    17: "Infield hit",
    18: "Line-drive out",
    19: "Error",
    20: "Hit by pitch",
    21: "Bunt foul",
    22: "Wild pitch",
    23: "Sacrifice bunt out",
    24: "Caught stealing",
    25: "Three-base hit",
    26: "Bunt hit",
    27: "Bunt out",
    28: "Passed ball",
    29: "Pickoff out",
}
LABEL_STR_TO_ID = {
    "Ball": 0,
    "Strike": 1,
    "Foul": 2,
    "Swing and a miss": 3,
    "Fly out": 4,
    "Ground out": 5,
    "One-base hit": 6,
    "Strike out": 7,
    "Home in": 8,
    "Base on balls": 9,
    "Touch out": 10,
    "Two-base hit": 11,
    "Homerun": 12,
    "Foul fly out": 13,
    "Double play": 14,
    "Tag out": 15,
    "Stealing base": 16,
    "Infield hit": 17,
    "Line-drive out": 18,
    "Error": 19,
    "Hit by pitch": 20,
    "Bunt foul": 21,
    "Wild pitch": 22,
    "Sacrifice bunt out": 23,
    "Caught stealing": 24,
    "Three-base hit": 25,
    "Bunt hit": 26,
    "Bunt out": 27,
    "Passed ball": 28,
    "Pickoff out": 29,
}


class BBDBDataset(Dataset):
    """
    PyTorch Dataset class for Baseball Database (BBDB).
    """

    def __init__(self, segment_filepaths, segment_length, frameskip, transform=None, meta_path="./bbdb.v0.9.min.json"):
        self.segment_length = segment_length
        self.frameskip = frameskip
        self.transform = transform

        self.segment_filepaths = segment_filepaths
        with open(meta_path) as fp:
            self.meta = json.load(fp)

        self.label_counts = np.zeros(NUM_LABELS)
        for segment_filepath in self.segment_filepaths:
            z = re.match("\./segments/(\w+)/(\w+).mp4", segment_filepath)
            # NOTE(seungjaeryanlee): segment_index is per video, not global
            gamecode, segment_index = z.groups()
            segment_index = int(segment_index)
            label_index = self.meta["database"][gamecode]["annotations"][segment_index]["labelIndex"]
            self.label_counts[label_index] += 1


    def __getitem__(self, index):
        """
        Return a segment with a given index.

        The segment has shape (channel, length, height, width).
        """
        segment_filepath = self.segment_filepaths[index]
        z = re.match("\./segments/(\w+)/(\w+).mp4", segment_filepath)
        # NOTE(seungjaeryanlee): segment_index is per video, not global
        gamecode, segment_index = z.groups()
        segment_index = int(segment_index)

        # Get metadata
        fps = self.meta["database"][gamecode]["fps"]
        label_index = self.meta["database"][gamecode]["annotations"][segment_index]["labelIndex"]

        data = skvideo.io.vread(segment_filepath)

        # NOTE(seungjaeryanlee): Some videos have 60fps, some have 30fps
        if fps > 30:
            data = data[::self.frameskip * 2]
        else:
            data = data[::self.frameskip]

        if self.transform is not None:
            data = self.transform(data)

        data = data[0:self.segment_length]

        # Normalize from [0, 255] to [-1, 1]
        data = data.astype(float) / 255. * 2 - 1

        # Change order of dimensions from (length, height, width, channel) to (channel, length, height, width)
        # NOTE(seungjaeryanlee): This must come after transforms
        data = data.transpose([3, 0, 1, 2])

        # Change label to onehot label per image
        onehot_label = np.zeros((NUM_LABELS, self.segment_length), dtype=np.float32)
        onehot_label[label_index, :] = 1

        return data, onehot_label

    def __len__(self):
        return len(self.segment_filepaths)


if __name__ == "__main__":
    with open("data_split.min.json", "r") as fp:
        data_split = json.load(fp)
    dataset = BBDBDataset(segment_filepaths=data_split["train"], segment_length=150, frameskip=1)
    print(dataset[0][0].shape)
    print(len(dataset))
