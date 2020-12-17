import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import argparse
import cv2, sys, os
from flownet2.networks import FlowNetS
sys.path.append('../')
from gaze_data import getDataset

if __name__ == "__main__":
    BATCH_SIZE = 1
    INPUT_CHANNELS = 6
    BATCH_NORM = False

    dataset = getDataset.ImageDataset('BookShelf_S1/')
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    a = iter(trainLoader)
    imgs = next(a)

    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)

    args = parser.parse_args()
    ## load model without batch norm
    checkpoint_path = "/home/sans/Downloads/gaze_data/FlowNet2-S_checkpoint.pth.tar"
    net = FlowNetS.FlowNetS(args, INPUT_CHANNELS, BATCH_NORM)
    dict = torch.load(checkpoint_path)
    net.load_state_dict(dict["state_dict"])
    newNet = nn.Sequential(*list(net.children())[0:10])
    newNet[9] = newNet[9][0]
    # x = x.view(-1, 17408x30)
    # net.append(nn.Linear(1024, 128))


    for params in newNet.parameters():
        params.requires_grad = False

    out = newNet(imgs)
    out = out.view(-1, 1024*17*30)
    vis_encoder = nn.Linear(1024*17*30, 1024)
    out = vis_encoder(out)
    print(out.shape)
