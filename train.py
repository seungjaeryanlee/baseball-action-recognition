import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
import video_transforms

import numpy as np
import wandb

from i3d import InceptionI3d
import bbdb_dataset


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_i3d(i3d, max_epoch, optimizer, lr_scheduler, dataloader, val_dataloader, save_model, num_steps_per_update, model_save_interval):

    train_batch_iterator = iter(dataloader)
    val_batch_iterator = iter(val_dataloader)

    # Training loop
    # Counted by number of minibatches, NOT number of updates
    steps = 0
    current_epoch = 0
    while current_epoch < max_epoch:
        # Training phase
        i3d.train(True)

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        num_iter = 0
        optimizer.zero_grad()

        for _ in range(num_steps_per_update):
            steps += 1
            try:
                inputs, labels = next(train_batch_iterator)
            except StopIteration:
                train_batch_iterator = iter(dataloader)
                inputs, labels = next(train_batch_iterator)
                current_epoch += 1
            inputs = inputs.float().cuda()
            labels = labels.cuda()
            t = inputs.size(2)

            per_frame_logits = i3d(inputs)
            # upsample to input size
            per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.item()

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()

            loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            tot_loss += loss.item()
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        wandb.log({ "lr": get_lr(optimizer) }, step=steps)
        print('Step {:5d} | Train | Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(
            steps,
            tot_loc_loss / num_steps_per_update,
            tot_cls_loss / num_steps_per_update,
            tot_loss,
        ))
        wandb.log({
            "train_loc_loss": tot_loc_loss / num_steps_per_update,
            "train_cls_loss": tot_cls_loss / num_steps_per_update,
            "train_tot_loss": tot_loss,
        }, step=steps)
        tot_loss = tot_loc_loss = tot_cls_loss = 0.

        # Validation phase
        i3d.train(False)

        with torch.no_grad():
            try:
                inputs, labels = next(val_batch_iterator)
            except StopIteration:
                val_batch_iterator = iter(val_dataloader)
                inputs, labels = next(val_batch_iterator)
            inputs = inputs.float().cuda()
            labels = labels.cuda()
            t = inputs.size(2)

            per_frame_logits = i3d(inputs)
            # upsample to input size
            per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

            # compute localization loss
            val_loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)

            # compute classification loss (with max-pooling along time B x C x T)
            val_cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])

            val_loss = 0.5 * val_loc_loss + 0.5 * val_cls_loss

            print('Step {:5d} | Valid | Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(
                steps,
                val_loc_loss.item(),
                val_cls_loss.item(),
                val_loss.item(),
            ))
            wandb.log({
                "val_loc_loss": val_loc_loss.item(),
                "val_cls_loss": val_cls_loss.item(),
                "val_tot_loss": val_loss.item(),
            }, step=steps)

        # Save model
        if steps % model_save_interval == 0:
            model_filename = save_model + str(steps).zfill(6) + '.pt'
            torch.save(i3d.module.state_dict(), model_filename)
            wandb.save(model_filename)

        # Update learning rate
        lr_scheduler.step(val_loss)

    # Save final model
    model_filename = save_model + str(steps).zfill(6) + '_final.pt'
    torch.save(i3d.module.state_dict(), model_filename)
    wandb.save(model_filename)


if __name__ == '__main__':
    CONFIG = {
        ## I3D
        "I3D_MODE": "rgb",
        "I3D_PRETRAINED_DATASET": "imagenet",
        "I3D_LOAD_MODEL_PATH": "",
        "I3D_SAVE_MODEL_PATH": "models/",

        ## Data
        "DATASET": "debug", # ["original", "binary"]
        "SEGMENT_LENGTH": 150,
        "FRAMESKIP": 1,

        ## Training
        "MAX_EPOCH": 1000,
        # NOTE(seungjaeryanlee): Originally 8*5, but lowered due to memory
        "BATCH_SIZE": 1,

        ## Learning Rate
        "INIT_LR": 0.1,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 0.0000001,

        ## Misc.
        # Accumulate gradient
        "NUM_STEPS_PER_UPDATE": 4,
        "MODEL_SAVE_INTERVAL": 1000,
    }
    assert CONFIG["SEGMENT_LENGTH"] * CONFIG["FRAMESKIP"] <= 150

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    CONFIG["I3D_SAVE_MODEL_PATH"] = CONFIG["I3D_SAVE_MODEL_PATH"] + timestamp + "/"
    Path(CONFIG["I3D_SAVE_MODEL_PATH"]).mkdir(parents=True, exist_ok=True)
    with open(CONFIG["I3D_SAVE_MODEL_PATH"] + "config.json", "w+") as fp:
        json.dump(CONFIG, fp, indent=4)

    # Setup wandb
    wandb.init(project="baseball-action-recognition", config=CONFIG)

    # Setup Datasets and Dataloaders
    if CONFIG["DATASET"] == "original":
        Dataset = bbdb_dataset.OriginalBBDBDataset
    elif CONFIG["DATASET"] == "binary":
        Dataset = bbdb_dataset.BinaryBBDBDataset
    elif CONFIG["DATASET"] == "debug":
        Dataset = bbdb_dataset.DebugBBDBDataset
    else:
        assert False

    with open("data_split.min.json", "r") as fp:
        data_split = json.load(fp)
    train_transforms = transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.RandomCrop(224),
        # TODO(seungjaeryanlee): Perhaps not needed
        video_transforms.RandomHorizontalFlip(),
    ])
    val_transforms = transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.CenterCrop(224),
    ])
    dataset = Dataset(
        segment_filepaths=data_split["train"],
        segment_length=CONFIG["SEGMENT_LENGTH"],
        frameskip=CONFIG["FRAMESKIP"],
        transform=train_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, pin_memory=True)

    val_dataset = Dataset(
        segment_filepaths=data_split["valid"],
        segment_length=CONFIG["SEGMENT_LENGTH"],
        frameskip=CONFIG["FRAMESKIP"],
        transform=val_transforms,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, pin_memory=True)

    # Setup I3D
    # Choose RGB-I3D or Flow-I3D
    if CONFIG["I3D_MODE"] == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('pretrained_models/{}_{}.pt'.format(CONFIG["I3D_MODE"], CONFIG["I3D_PRETRAINED_DATASET"])))
    i3d.replace_logits(dataset.NUM_LABELS)
    if CONFIG["I3D_LOAD_MODEL_PATH"]:
        i3d.load_state_dict(torch.load(CONFIG["I3D_LOAD_MODEL_PATH"]))
    i3d = i3d.cuda()
    i3d = nn.DataParallel(i3d)

    # Setup optimizer and lr_scheduler
    optimizer = optim.SGD(
        i3d.parameters(),
        lr=CONFIG["INIT_LR"],
        momentum=CONFIG["MOMENTUM"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
    )
    lr_scheduler = ReduceLROnPlateau(optimizer)

    train_i3d(
        i3d=i3d,
        max_epoch=CONFIG["MAX_EPOCH"],
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        save_model=CONFIG["I3D_SAVE_MODEL_PATH"],
        num_steps_per_update=CONFIG["NUM_STEPS_PER_UPDATE"],
        model_save_interval=CONFIG["MODEL_SAVE_INTERVAL"],
    )
