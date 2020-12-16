import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import argparse
from flownet2.networks import FlowNetS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)

    args = parser.parse_args()
    ## load model without batch norm
    net = FlowNetS.FlowNetS(args, input_channels=6, batchNorm=False)
    # net = nn.Sequential(*net)
    # net = list(net.children())[0:9]
    # newFlowNet = nn.Sequential(*net)
    dict = torch.load("/home/sans/Downloads/gaze_data/FlowNet2-S_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    newNet = nn.Sequential(*list(net.children())[0:9])
    # print(newNet)
    for params in net.parameters():
        params.requires_grad = False
