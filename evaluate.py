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


def evaluate_i3d(i3d, dataset, dataloader):
    correct_count = 0
    all_predictions = []
    all_labels = []
    i3d.train(False)
    for inputs, labels in tqdm(dataloader):
        with torch.no_grad():
            inputs = inputs.float().cuda()
            labels = labels.cuda()
            t = inputs.size(2)

            per_frame_logits = i3d(inputs)

            # For prediction, use the last frame prediction only
            predictions = per_frame_logits.max(dim=1)[1][:, -1]
            # One-hot to index
            labels = (labels[:,:,0] == 1).nonzero()[:, 1]

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.long().cpu().numpy())
            correct_count += (predictions == labels.long()).sum().item()

        print("Current Accuracy: {:.4f} = {}/{}".format(
            correct_count / (len(all_predictions) * len(all_predictions[0])),
            correct_count,
            len(all_predictions) * len(all_predictions[0])
        ))

    return (
        correct_count / len(dataset),
        np.concatenate(all_predictions),
        np.concatenate(all_labels),
    )


if __name__ == '__main__':
    CONFIG = {
        ## I3D
        "RGB_I3D_LOAD_MODEL_PATH": "models/20200110-054139/004000.pt",
        # TODO(seungjaeryanlee): Flow I3D Not yet integrated
        "FLOW_I3D_LOAD_MODEL_PATH": "",

        ## Data
        "SEGMENT_LENGTH": 37,
        "FRAMESKIP": 4,

        ## Training
        # NOTE(seungjaeryanlee): Originally 8*5, but lowered due to memory
        "BATCH_SIZE": 4,
    }
    assert CONFIG["RGB_I3D_LOAD_MODEL_PATH"] or CONFIG["FLOW_I3D_LOAD_MODEL_PATH"]

    # Setup I3D
    # TODO(seungjaeryanlee): Allow choosing both
    if CONFIG["RGB_I3D_LOAD_MODEL_PATH"]:
        rgb_i3d = InceptionI3d(400, in_channels=3)
        rgb_i3d.replace_logits(bbdb_dataset.NUM_LABELS)
        rgb_i3d.load_state_dict(torch.load(CONFIG["RGB_I3D_LOAD_MODEL_PATH"]))
        rgb_i3d = rgb_i3d.cuda()
        # TODO(seungjaeryanlee): Not needed?
        rgb_i3d = nn.DataParallel(rgb_i3d)

    # Setup Dataset and Dataloader
    with open("data_split.min.json", "r") as fp:
        data_split = json.load(fp)
    test_transforms = transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.CenterCrop(224),
    ])
    dataset = bbdb_dataset.BBDBDataset(
        segment_filepaths=data_split["test"],
        segment_length=CONFIG["SEGMENT_LENGTH"],
        frameskip=CONFIG["FRAMESKIP"],
        transform=test_transforms
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], pin_memory=True)

    accuracy, predictions, labels = evaluate_i3d(i3d=rgb_i3d, dataset=dataset, dataloader=dataloader)

    with open(CONFIG["RGB_I3D_LOAD_MODEL_PATH"].replace(".pt", ".json"), "w+") as fp:
        json.dump({
            "accuracy": accuracy,
            "predictions": predictions.tolist(),
            "labels": labels.tolist(),
        }, fp)
