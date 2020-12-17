import torch
from torch.utils.data import Dataset
import pandas as pd
import sys, ast
import numpy as np

class IMUDataset(Dataset):
    def __init__(self, rootfolder, device=None):
        self.root = '/home/sans/Downloads/gaze_data/'
        self.rootfolder = rootfolder
        self.path = self.root + self.rootfolder  + '/' if self.rootfolder[-1]!='/' else (self.root + self.rootfolder)
        self.data = pd.read_csv(self.path + 'imu_file.csv').T
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imu_values = self.data.iloc[:, index][1:]
        data_pts = []
        for index, data in enumerate(imu_values):
            try:
                (acc, gyro) = ast.literal_eval(data)
                data_pt = np.array(acc + gyro)
                data_pt[1] += 9.80665
                data_pts.append(np.round(data_pt, 3))
            except Exception as e:
                print(e)

        return np.transpose(np.array(data_pts))

BATCH_SIZE = 1
folder = sys.argv[1]
dataset = IMUDataset(folder)
trainLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
a = iter(trainLoader)
data = next(a)
print(data.shape, data)
