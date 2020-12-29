import torch
from torch.utils.data import Dataset
import pandas as pd
import sys, ast, os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pickle as pkl
sys.path.append('../')
from variables import Variables, RootVariables
from build_dataset import BUILDING_DATASETS
from torchvision import transforms

## save the extracted dataset in each folder.
class UNIFIED_DATASET(Dataset):
    def __init__(self, frame_data, imu_data, gaze_data, device=None):
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.frame_data = frame_data
        self.imu_data = imu_data
        self.gaze_data = gaze_data
        self.device = device

    def __len__(self):
        return len(self.imu_data) -1

    def __getitem__(self, index):
        # return self.transforms(elf.frame_data[index]).to(self.device), torch.from_numpy(np.concatenate((self.imu_data[index], self.imu_data[index+1]), axis=0)).to(self.device), torch.from_numpy(self.gaze_data[index]).to(self.device)
        return self.transforms(self.frame_data[index]).to(self.device), torch.from_numpy(np.concatenate((self.imu_data[index], self.imu_data[index+1]), axis=0)).to(self.device), torch.from_numpy(np.concatenate((self.gaze_data[index], self.gaze_data[index+1]), axis=0)).to(self.device)


class IMU_GAZE_FRAME_DATASET:
    def __init__(self, root, trim_size, distribution='S'):
        self.root = root
        self.dataset = BUILDING_DATASETS(self.root, trim_size)
        self.frame_datasets = None
        self.imu_datasets, self.gaze_datasets = None, None
        if Path(self.root + 'imuExtracted_data_' + str(trim_size) + '.pt').is_file():
            print('Files exists')
            self.imu_datasets = torch.load('imuExtracted_data_' + str(trim_size) + '.pt')
            self.gaze_datasets = torch.load('gazeExtracted_data_' + str(trim_size) + '.pt')
        else:
            print('saved files does not exis')
            self.imu_datasets = self.dataset.load_unified_imu_dataset()
            self.gaze_datasets = self.dataset.load_unified_gaze_dataset()
            self.imu_datasets = torch.from_numpy(self.imu_datasets)
            self.gaze_datasets = torch.from_numpy(self.gaze_datasets)
            torch.save(self.imu_datasets, self.root + 'imuExtracted_data_' + str(trim_size) + '.pt')
            torch.save(self.gaze_datasets, self.root + 'gazeExtracted_data_' + str(trim_size) + '.pt')

        self.frame_datasets = self.dataset.load_unified_frame_dataset()

        if distribution == 'N':
            self.imu_datasets = self.dataset.normalization(self.imu_datasets)
        else:
            self.imu_datasets = self.dataset.standarization(self.imu_datasets)

        #self.gaze_datasets = self.dataset.normalization(self.gaze_datasets)

        self.gaze_datasets = self.gaze_datasets.reshape(-1, 4, self.gaze_datasets.shape[-1])
        self.imu_datasets = self.imu_datasets.reshape(-1, 4, self.imu_datasets.shape[-1])

    def __len__(self):
        return int(len(self.imu_datasets))      ## number of frames corresponding to

    # def __getitem__(self, index):
    #     return self.imu_datasets[index], self.gaze_datasets[index]

if __name__ =="__main__":
    var = RootVariables()
    device = torch.device("cpu")
    datasets = IMU_GAZE_FRAME_DATASET(var.root, 150, device)
    s = 0
    e = 1200
    imu = datasets.get_imu_gaze_datas()
    imudata = imu[:100]
    data = torch.utils.data.DataLoader(imudata, batch_size=5)
    a = iter(data)
    x = next(a)
    print(len(data), a)
