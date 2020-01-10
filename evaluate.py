import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import video_transforms

from i3d import InceptionI3d
import bbdb_dataset


def evaluate_i3d(i3d, dataloader):
    i3d.train(False)
    for inputs, labels in dataloader:
        with torch.no_grad():
            inputs = inputs.float().cuda()
            labels = labels.cuda()
            t = inputs.size(2)

            per_frame_logits = i3d(inputs)
            # TODO(seungjaeryanlee): Compute accuracy

            # upsample to input size
            per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])

            loss = 0.5 * val_loc_loss + 0.5 * val_cls_loss


if __name__ == '__main__':
    CONFIG = {
        ## I3D
        "RGB_I3D_LOAD_MODEL_PATH": "",
        # TODO(seungjaeryanlee): Flow I3D Not yet integrated
        "FLOW_I3D_LOAD_MODEL_PATH": "",

        ## Data
        "FRAMESKIP": 1, # TODO(seungjaeryanlee): 1, 4, ?

        ## Training
        # NOTE(seungjaeryanlee): Originally 8*5, but lowered due to memory
        "BATCH_SIZE": 4,
    }
    assert CONFIG["I3D_USE_RGB"] or CONFIG["I3D_USE_FLOW"]

    # Setup I3D
    # TODO(seungjaeryanlee): Allow choosing both
    if CONFIG["RGB_I3D_LOAD_MODEL_PATH"]:
        rgb_i3d = InceptionI3d(400, in_channels=3)
        rgb_i3d.load_state_dict(torch.load(CONFIG["I3D_LOAD_MODEL_PATH"]))
        # TODO(seungjaeryanlee): Is this not on GPU?
        rgb_i3d.cuda()
        # TODO(seungjaeryanlee): Not needed?
        rgb_i3d = nn.DataParallel(i3d)

    # Setup Dataset and Dataloader
    with open("data_split.min.json", "r") as fp:
        data_split = json.load(fp)
    test_transforms = transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.CenterCrop(224),
    ])
    dataset = bbdb_dataset.BBDBDataset(segment_filepaths=data_split["test"], frameskip=CONFIG["FRAMESKIP"], transform=test_transforms)
    dataloader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, pin_memory=True)

    evaluate_i3d(i3d=i3d, dataloader=dataloader)
