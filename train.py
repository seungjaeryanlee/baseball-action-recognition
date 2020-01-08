import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
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


def train_i3d(i3d, max_steps, optimizer, lr_scheduler, dataloader, val_dataloader, save_model):
    dataloaders = { 'train': dataloader, 'val': val_dataloader }
    # Training loop
    num_steps_per_update = 4 # Accumulate gradient
    steps = 0
    while steps < max_steps: # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('----------')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            for data in dataloaders[phase]:
                num_iter += 1

                inputs, labels = data
                inputs = inputs.float().cuda()
                t = inputs.size(2)
                labels = labels.cuda()

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

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    wandb.log({ "lr": get_lr(optimizer) }, step=steps)
                    if steps % 10 == 0:
                        # Save model
                        model_filename = save_model + str(steps).zfill(6) + '.pt'
                        torch.save(i3d.module.state_dict(), model_filename)
                        wandb.save(model_filename)
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        wandb.log({
                            "train_loc_loss": tot_loc_loss / (10 * num_steps_per_update),
                            "train_cls_loss": tot_cls_loss / (10 * num_steps_per_update),
                            "train_tot_loss": tot_loss / 10,
                        }, step=steps)
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
                wandb.log({
                    "val_loc_loss": tot_loc_loss/num_iter,
                    "val_cls_loss": tot_cls_loss/num_iter,
                    "val_tot_loss": (tot_loss * num_steps_per_update) / num_iter,
                }, step=steps)


if __name__ == '__main__':
    CONFIG = {
        ## I3D
        "I3D_MODE": "rgb",
        "I3D_PRETRAINED_DATASET": "imagenet",
        "I3D_LOAD_MODEL_PATH": "",
        "I3D_SAVE_MODEL_PATH": "models/",

        ## Data
        "FRAMESKIP": 1,

        ## Training
        "MAX_STEPS": 6400,
        # NOTE(seungjaeryanlee): Originally 8*5, but lowered due to memory
        "BATCH_SIZE": 4,

        ## Learning Rate
        "INIT_LR": 0.1,
        "MULTISTEP_LR_MILESTONES": [300, 1000],
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 0.0000001,

        ## Misc.
    }

    # Setup wandb
    wandb.init(project="baseball-action-recognition", config=CONFIG)

    # Setup I3D
    # Choose RGB-I3D or Flow-I3D
    if CONFIG["I3D_MODE"] == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('pretrained_models/{}_{}.pt'.format(CONFIG["I3D_MODE"], CONFIG["I3D_PRETRAINED_DATASET"])))
    i3d.replace_logits(bbdb_dataset.NUM_LABELS)
    if CONFIG["I3D_LOAD_MODEL_PATH"]:
        i3d.load_state_dict(torch.load(CONFIG["I3D_LOAD_MODEL_PATH"]))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    # Setup optimizer and lr_scheduler
    optimizer = optim.SGD(
        i3d.parameters(),
        lr=CONFIG["INIT_LR"],
        momentum=CONFIG["MOMENTUM"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
    )
    lr_scheduler = MultiStepLR(optimizer, CONFIG["MULTISTEP_LR_MILESTONES"])

    # Setup Datasets and Dataloaders
    # TODO(seungjaeryanlee): Setup transforms (Rescale, RandomCrop, ToTensor)
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    with open("data_split.min.json", "r") as fp:
        data_split = json.load(fp)
    train_transforms = transforms.Compose([
        video_transforms.RandomCrop(224),
        video_transforms.RandomHorizontalFlip(),
    ])
    val_transforms = transforms.Compose([
        video_transforms.CenterCrop(224),
    ])
    dataset = bbdb_dataset.BBDBDataset(segment_filepaths=data_split["train"], frameskip=CONFIG["FRAMESKIP"], transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, pin_memory=True)

    val_dataset = bbdb_dataset.BBDBDataset(segment_filepaths=data_split["valid"], frameskip=CONFIG["FRAMESKIP"], transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, pin_memory=True)

    train_i3d(
        i3d=i3d,
        max_steps=CONFIG["MAX_STEPS"],
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        save_model=CONFIG["I3D_SAVE_MODEL_PATH"],
    )
