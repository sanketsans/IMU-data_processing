import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys, os, ast
sys.path.append('../')
from variables import RootVariables

device = torch.device("cpu")

class IMU_ENCODER(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(IMU_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True).to(self.device)
        self.fc = nn.Linear(hidden_size*2, num_classes).to(self.device)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.randn(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        # hidden = (h0, c0)
        out, _ = self.lstm(x, (h0, c0))
        #out = F.relu(self.dropout(self.fc(out[:, -1, :])))

        return out[:,-1,:]

class TEMP_ENCODER(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(TEMP_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True).to(self.device)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)

        return out.to(self.device), hidden

## PREPARING THE DATA
# folder = sys.argv[1]
# dataset_folder = '/home/sans/Downloads/gaze_data/'
# os.chdir(dataset_folder + folder + '/' if folder[-1]!='/' else (dataset_folder + folder))
if __name__ == "__main__":
    folder = sys.argv[1]
    device = torch.device("cpu")

    var = RootVariables()
    os.chdir(var.root + folder)
    dataset = FRAME_IMU_DATASET(var.root, folder, 150, device)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=var.batch_size, drop_last=True)
    a = iter(trainLoader)
    f, g, i = next(a)
    # print(data.shape, data)
    print(i.shape) # [batch_size, sequence_length, input_size]
    i = i.reshape(i.shape[0], i.shape[2], -1)
    print(i.shape)

    model = IMU_ENCODER(var.input_size, var.hidden_size, var.num_layers, var.num_classes).to(device)
    scores = model(data.float())
    print(model, scores.shape)
    scores = scores.unsqueeze(dim = 1)
    newscore = scores.reshape(scores.shape[0], 4, 32)
    print(newscore.shape)
    print(newscore)
