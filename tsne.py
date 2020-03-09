"""
Generate t-SNE plot using embeddings from a trained model.
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
from sklearn.manifold import TSNE


def get_embeddings(i3d, dataset, dataloader):
    correct_count = 0
    all_predictions = []
    all_labels = []
    i3d.train(False)
    embeddings = []
    for inputs, labels in tqdm(dataloader):
        with torch.no_grad():
            inputs = inputs.float().cuda()
            labels = labels.cuda()
            t = inputs.size(2)

            embedding = i3d.get_embedding(inputs)
            # NOTE(seungjaeryanlee): Use last frame only
            embedding = embedding[:, :, -1, 0, 0]
            embeddings.append(embedding.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

if __name__ == '__main__':
    CONFIG = {
        ## I3D
        "RGB_I3D_LOAD_MODEL_PATH": "models/20200113-100601/002240_final.pt",
        # TODO(seungjaeryanlee): Flow I3D Not yet integrated
        "FLOW_I3D_LOAD_MODEL_PATH": "",

        ## Data
        "DATASET": "binary",
        "SEGMENT_LENGTH": 150,
        "FRAMESKIP": 1,

        ## Training
        # NOTE(seungjaeryanlee): Originally 8*5, but lowered due to memory
        "BATCH_SIZE": 4,
    }
    assert CONFIG["RGB_I3D_LOAD_MODEL_PATH"] or CONFIG["FLOW_I3D_LOAD_MODEL_PATH"]

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
    dataloader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], pin_memory=True)

    # Setup I3D
    # TODO(seungjaeryanlee): Allow choosing both
    if CONFIG["RGB_I3D_LOAD_MODEL_PATH"]:
        rgb_i3d = InceptionI3d(400, in_channels=3)
        rgb_i3d.replace_logits(dataset.NUM_LABELS)
        rgb_i3d.load_state_dict(torch.load(CONFIG["RGB_I3D_LOAD_MODEL_PATH"]))
        rgb_i3d = rgb_i3d.cuda()
        # TODO(seungjaeryanlee): Not needed?
        # rgb_i3d = nn.DataParallel(rgb_i3d)


    with open("models/20200110-054139/004000.json", "r") as fp:
        original_labels = json.load(fp)["labels"]

    original_labels = np.array(original_labels)
    original_labels = original_labels[(original_labels != 16) & (original_labels != 24) & (original_labels != 29)]

    with open("models/20200113-100601/002240_final.json", "r") as fp:
        data = json.load(fp)
        binary_predictions = np.asarray(data["predictions"])
        binary_labels = np.asarray(data["labels"])


    try:
        embeddings = np.load("embeddings.npy")
        embeddings_2d = np.load("embeddings_2d.npy")
    except IOError:
        embeddings = get_embeddings(rgb_i3d, dataset, dataloader)
        np.save("embeddings.npy", embeddings)
        embeddings_2d = TSNE(n_components=2).fit_transform(np.asarray(embeddings))
        np.save("embeddings_2d.npy", embeddings_2d)


    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    batting_indices = (binary_labels == 1).nonzero()[0]
    no_hit_indices = (binary_labels == 0).nonzero()[0]
    # batting_indices = batting_indices[batting_indices < len(embeddings_2d)]
    # no_hit_indices = no_hit_indices[no_hit_indices < len(embeddings_2d)]
    axs[0].set_title("Ground Truth")
    axs[0].scatter(embeddings_2d[batting_indices, 0], embeddings_2d[batting_indices, 1], c="blue")
    axs[0].scatter(embeddings_2d[no_hit_indices, 0], embeddings_2d[no_hit_indices, 1], c="green")
    axs[0].legend(["Batting", "No hit"])

    predicted_batting_indices = (binary_predictions == 1).nonzero()[0]
    predicted_no_hit_indices = (binary_predictions == 0).nonzero()[0]
    axs[1].set_title("Prediction")
    axs[1].scatter(embeddings_2d[predicted_batting_indices, 0], embeddings_2d[predicted_batting_indices, 1], c="blue")
    axs[1].scatter(embeddings_2d[predicted_no_hit_indices, 0], embeddings_2d[predicted_no_hit_indices, 1], c="green")
    axs[1].legend(["Batting", "No hit"])

    axs[2].set_title("Ball, Strike, and Foul")
    colors = plt.cm.hsv(np.linspace(0.1, 0.9, 3))
    for i, color in enumerate(colors):
        if i not in [0, 1, 2]: continue
        indices = (original_labels == i).nonzero()[0]
        axs[2].scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=[colors[i]])
    axs[2].legend([bbdb_dataset.LABEL_ID_TO_STR[i] for i in range(3)])

    plt.show()
