import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys, os, ast
sys.path.append('../')
from gaze_data import getDataset


input_size = 6
hidden_size = 128
num_layers = 2
sequence_size = 4
num_classes = 128
device = torch.device("cpu")

class BILSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BILSTM, self).__init__()
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
BATCH_SIZE = 1
folder = sys.argv[1]
dataset = getDataset.IMUDataset(folder)
trainLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
a = iter(trainLoader)
data = next(a)
# print(data.shape, data)
print(data.shape)

model = BILSTM(input_size, hidden_size, num_layers, num_classes).to(device)
scores = model(data.float())
print(model, scores.shape)
