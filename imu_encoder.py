import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys, os, ast
sys.path.append('../')
from getDataset import IMUDataset
from variables import RootVariables

device = torch.device("cpu")

class IMU_ENCODER(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(IMU_ENCODER, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

## PREPARING THE DATA
# folder = sys.argv[1]
# dataset_folder = '/home/sans/Downloads/gaze_data/'
# os.chdir(dataset_folder + folder + '/' if folder[-1]!='/' else (dataset_folder + folder))
if __name__ == "__main__":
    folder = sys.argv[1]
    var = RootVariables()
    dataset = IMUDataset(var.root, folder)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=var.batch_size)
    a = iter(trainLoader)
    data = next(a)
    # print(data.shape, data)
    print(data.shape) # [batch_size, sequence_length, input_size]

    model = IMU_ENCODER(var.input_size, var.hidden_size, var.num_layers, var.num_classes).to(device)
    scores = model(data.float())
    print(model, scores.shape)
