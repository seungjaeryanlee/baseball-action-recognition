import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
import video_transforms

import numpy as np

from i3d import InceptionI3d
import bbdb_dataset


def train_i3d(i3d, optimizer, init_lr, lr_scheduler, dataloader, val_dataloader, max_steps=64e3, save_model=''):
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
            
            # Iterate over data.
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

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.item()
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        # save model
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    #os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-save_model', type=str)

    args = parser.parse_args()

    # Setup I3D
    # Choose RGB-I3D or Flow-I3D
    if args.mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(bbdb_dataset.NUM_LABELS)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    # Setup optimizer and lr_scheduler
    init_lr = 0.1
    optimizer = optim.SGD(i3d.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    # Setup Datasets and Dataloaders
    # NOTE(seungjaeryanlee): Originally 8*5, but lowered due to memory
    batch_size = 4
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
    # TODO(seungjaeryanlee): Check num_workers for DataLoader
    dataset = bbdb_dataset.BBDBDataset(segment_filepaths=data_split["train"], frameskip=1, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_dataset = bbdb_dataset.BBDBDataset(segment_filepaths=data_split["valid"], frameskip=1, transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    train_i3d(
        i3d,
        optimizer,
        init_lr,
        lr_scheduler,
        dataloader,
        val_dataloader,
        save_model=args.save_model if args.save_model else ""
    )
