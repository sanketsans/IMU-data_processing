import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys, os, ast
# sys.path.append('../')

input_size = 6
hidden_size = 128
num_layers = 2
sequence_size = 4

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
folder = sys.argv[1]
dataset_folder = '/home/sans/Downloads/gaze_data/'
os.chdir(dataset_folder + folder + '/' if folder[-1]!='/' else (dataset_folder + folder))

df_imu = pd.read_csv('imu_file.csv').T

data_pts = df_imu.iloc[:, 10]
for index, data in enumerate(data_pts):
    try:
        (acc, gyro) = ast.literal_eval(data)
        data_pt = np.array(acc + gyro)
        data_pt[1] += 9.80665
        data_pt = np.round(data_pt, 3)
        # data_pt = data_pt.reshape(6, 1)
        # print(acc[1] + 9.8)
        print(data_pt, data_pt.shape )
    except Exception as e:
        print(e)
