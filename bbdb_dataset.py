"""
Define the PyTorch Dataset class for Baseball Database (BBDB).
"""
import skvideo.io
import glob
import json
import re

from torch.utils.data import Dataset


class BBDBDataset(Dataset):
    """
    PyTorch Dataset class for Baseball Database (BBDB).
    """

    def __init__(self, frameskip, meta_path="./bbdb.v0.9.min.json"):
        self.frameskip = frameskip

        self.segment_filepaths = glob.glob("./segments/**/*.mp4")
        with open(meta_path) as fp:
            self.meta = json.load(fp)

    def __getitem__(self, index):
        """
        Return a segment with a given index.

        The segment has shape (length, height, width, channel).
        """
        segment_filepath = self.segment_filepaths[index]
        z = re.match("\./segments/(\w+)/(\w+).mp4", segment_filepath)
        # NOTE(seungjaeryanlee): segment_index is per video, not global
        gamecode, segment_index = z.groups()
        segment_index = int(segment_index)

        # Get metadata
        fps = self.meta["database"][gamecode]["fps"]
        label_index = self.meta["database"][gamecode]["annotations"][segment_index][
            "labelIndex"
        ]

        data = skvideo.io.vread(segment_filepath)

        # NOTE(seungjaeryanlee): Some videos have 60fps, some have 30fps
        if fps > 30:
            return data[:: self.frameskip * 2], label_index
        else:
            return data[:: self.frameskip], label_index

    def __len__(self):
        return len(self.segment_filepaths)


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


if __name__ == "__main__":
    dataset = BBDBDataset(frameskip=1)
    print(dataset[0][0].shape)
