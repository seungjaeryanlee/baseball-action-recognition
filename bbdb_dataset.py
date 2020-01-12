"""
Define the PyTorch Dataset class for Baseball Database (BBDB).
"""
import glob
import json
import re

import numpy as np
import skvideo.io
from torch.utils.data import Dataset


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

    def __init__(self, label_modifier, segment_filepaths, segment_length, frameskip, transform=None, meta_path="./bbdb.v0.9.min.json"):
        self.segment_length = segment_length
        self.frameskip = frameskip
        self.transform = transform

        self.label_modifier = label_modifier
        self.NUM_LABELS = len(set(label_modifier.values()))
        if None in set(label_modifier.values()):
            self.NUM_LABELS -= 1

        with open(meta_path) as fp:
            self.meta = json.load(fp)

        self.segment_filepaths = []
        self.label_counts = np.zeros(self.NUM_LABELS)
        for segment_filepath in segment_filepaths:
            z = re.match("\./segments/(\w+)/(\w+).mp4", segment_filepath)
            # NOTE(seungjaeryanlee): segment_index is per video, not global
            gamecode, segment_index = z.groups()
            segment_index = int(segment_index)
            original_label_index = self.meta["database"][gamecode]["annotations"][segment_index]["labelIndex"]
            label_index = self.label_modifier[original_label_index]
            if label_index is not None:
                self.label_counts[label_index] += 1
                self.segment_filepaths.append(segment_filepath)


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
        original_label_index = self.meta["database"][gamecode]["annotations"][segment_index]["labelIndex"]
        label_index = self.label_modifier[original_label_index]

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
        onehot_label = np.zeros((self.NUM_LABELS, self.segment_length), dtype=np.float32)
        onehot_label[label_index, :] = 1

        return data, onehot_label

    def __len__(self):
        return len(self.segment_filepaths)


class OriginalBBDBDataset(BBDBDataset):
    def __init__(self, segment_filepaths, segment_length, frameskip, transform=None, meta_path="./bbdb.v0.9.min.json"):
        super().__init__(
            label_modifier={ i: i for i in range(30) },
            segment_filepaths=segment_filepaths,
            segment_length=segment_length,
            frameskip=frameskip,
            transform=transform,
            meta_path=meta_path,
        )


class BinaryBBDBDataset(BBDBDataset):
    def __init__(self, segment_filepaths, segment_length, frameskip, transform=None, meta_path="./bbdb.v0.9.min.json"):
        binary_label_modifier = {
            0: 0, # "Ball": "No hit",
            1: 0, # "Strike": "No hit",
            2: 1, # "Foul": "Batting",
            3: 0, # "Swing and a miss": "No hit",
            4: 1, # "Fly out": "Batting",
            5: 1, # "Ground out": "Batting",
            6: 1, # "One-base hit": "Batting",
            7: 0, # "Strike out": "No hit",
            8: 1, # "Home in": "Batting",
            9: 0, # "Base on balls": "No hit",
            10: 1, # "Touch out": "Batting",
            11: 1, # "Two-base hit": "Batting",
            12: 1, # "Homerun": "Batting",
            13: 1, # "Foul fly out": "Batting",
            14: 1, # "Double play": "Batting",
            15: 1, # "Tag out": "Batting",
            16: None, # "Stealing base": None,
            17: 1, # "Infield hit": "Batting",
            18: 1, # "Line-drive out": "Batting",
            19: 1, # "Error": "Batting",
            20: 0, # "Hit by pitch": "No hit",
            21: 1, # "Bunt foul": "Batting",
            22: 0, # "Wild pitch": "No hit",
            23: 1, # "Sacrifice bunt out": "Batting",
            24: None, # "Caught stealing": None,
            25: 1, # "Three-base hit": "Batting",
            26: 1, # "Bunt hit": "Batting",
            27: 1, # "Bunt out": "Batting",
            28: 0, # "Passed ball": "No hit",
            29: None, # "Pickoff out": None,
        }

        super().__init__(
            label_modifier=binary_label_modifier,
            segment_length=segment_length,
            segment_filepaths=segment_filepaths,
            frameskip=frameskip,
            transform=transform,
            meta_path=meta_path,
        )


if __name__ == "__main__":
    with open("data_split.min.json", "r") as fp:
        data_split = json.load(fp)
    dataset = OriginalBBDBDataset(segment_filepaths=data_split["train"], segment_length=150, frameskip=1)
    print(dataset.label_counts)
    print(len(dataset))

    dataset = BinaryBBDBDataset(segment_filepaths=data_split["train"], segment_length=150, frameskip=1)
    print(dataset.label_counts)
    print(len(dataset))
