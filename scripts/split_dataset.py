"""
Script to split the dataset into train-valid-test split.
"""
import glob
import json
import random


def train_valid_test_split(data, valid_ratio=0.2, test_ratio=0.2):
    data = data.copy()
    random.shuffle(data)
    count = len(data)

    valid_data = data[:int(count * valid_ratio)]
    test_data = data[int(count * valid_ratio):int(count * (test_ratio + valid_ratio))]
    train_data = data[int(count * (test_ratio + valid_ratio)):]

    return train_data, valid_data, test_data


if __name__ == "__main__":
    random.seed(42)
    segment_filepaths = glob.glob("./segments/**/*.mp4")
    train_segment_filepaths, valid_segment_filepaths, test_segment_filepaths = train_valid_test_split(segment_filepaths, valid_ratio=0.2, test_ratio=0.2)

    # Save to JSON
    with open("data_split.min.json", "w+") as fp:
        json.dump({
            "train": train_segment_filepaths,
            "valid": valid_segment_filepaths,
            "test": test_segment_filepaths,
        }, fp)
