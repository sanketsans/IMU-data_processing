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
from getDataset import ImageDataset
from variables import RootVariables

class VIS_ENCODER:
    def __init__(self, args, checkpoint_path, device, input_channels=6, batch_norm=False):
        self.net = FlowNetS.FlowNetS(args, input_channels, batch_norm).to(device)
        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.newNet = nn.Sequential(*list(self.net.children())[0:10])
        self.newNet[9] = self.newNet[9][0]

        for params in self.newNet.parameters():
            params.requires_grad = False

    def run_model(self, input_img):
        out = self.newNet(input_img)
        out = out.view(-1, 8192*8)
        self.fc = nn.Linear(8192*8, 1024)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    var = RootVariables()
    device = torch.device("cpu")
    dataset = ImageDataset(var.root, 'imu_BookShelf_S1/', device)
    dataset.populate_data(dataset.first_frame)
    print(var.batch_size, len(dataset))
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=var.batch_size)
    a = iter(trainLoader)
    imgs = next(a)
    print(imgs.shape)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)

    args = parser.parse_args()
    ## load model without batch norm
    checkpoint_path = var.root + "FlowNet2-S_checkpoint.pth.tar"

    img_enc = VIS_ENCODER(args, checkpoint_path, device)
    output = img_enc.run_model(imgs)
