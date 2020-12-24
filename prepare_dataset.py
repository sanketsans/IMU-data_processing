import torch
from torch.utils.data import Dataset
from torchvision import transforms
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

## save the extracted dataset in each folder.
class IMU_GAZE_DATASET(Dataset):
    def __init__(self, root, trim_size, device = None, distribution='S'):
        self.root = root
        self.device = device
        self.dataset = BUILDING_DATASETS(self.root, trim_size)
        self.frame_datasets = None
        self.imu_datasets, self.gaze_datasets = None, None
        if Path(self.root + 'framesExtracted_data.npy').is_file():
            print('Files exists')
            with open(self.root + 'imuExtracted_data.npy', 'rb') as f:
                self.imu_datasets = np.load(f)
            with open(self.root + 'gazeExtracted_data.npy', 'rb') as f:
                self.gaze_datasets = np.load(f)
        else:
            print('saved files does not exis')
            self.imu_datasets = self.dataset.load_unified_imu_dataset()
            self.gaze_datasets= self.dataset.load_unified_gaze_dataset()
            self.frame_datasets = self.dataset.load_unified_frame_dataset('imu_')
            with open(self.root + 'imuExtracted_data.npy', 'wb') as f:
                np.save(f, self.imu_datasets)
            with open(self.root + 'gazeExtracted_data.npy', 'wb') as f:
                np.save(f, self.gaze_datasets)

        if distribution == 'N':
            self.imu_datasets = self.dataset.normalization(self.imu_datasets)
        else:
            self.imu_datasets = self.dataset.standarization(self.imu_datasets)

        self.gaze_datasets = self.gaze_datasets.reshape(-1, 4, self.gaze_datasets.shape[-1])
        self.imu_datasets = self.imu_datasets.reshape(-1, 4, self.imu_datasets.shape[-1])
        print(self.imu_datasets.shape, self.gaze_datasets.shape)
        self.gaze_pts_stack = None
        self.imu_pts_stack = None

    def __len__(self):
        return int(len(self.imu_datasets))      ## number of frames corresponding to

    def __getitem__(self, index):
        self.gaze_pts_stack, self.imu_pts_stack = [], []
        for i in range(2):
            # self.gaze_pts_stack.append(self.gaze_datasets[index+i])
            self.imu_pts_stack.append(self.imu_datasets[index+i])

        return torch.tensor(self.imu_pts_stack).to(self.device)#, torch.tensor(self.gaze_pts_stack).to(self.device)

class FRAME_DATASET(Dataset):
    def __init__(self, root, subDir, trim_size, device=None):
        self.root = root
        self.rootfolder = subDir + '/' if subDir[-1]!='/' else  subDir
        self.path = self.root + self.rootfolder + 'scenevideo.mp4'
        self.trim_size = trim_size
        self.device = device
        self.stack_frames = None
        self.last_frame, self.new_frame = None, None
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.capture = cv2.VideoCapture(self.path)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.frame_count - self.trim_size*2

    def __getitem__(self, index):
        if ((self.trim_size + index) < (self.frame_count - self.trim_size - 1)):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.trim_size+ index)
            for i in range(2):
                self.ret, self.new_frame = self.capture.read()
                self.new_frame = cv2.resize(self.new_frame, (512, 512))
                if i == 0:
                    self.last_frame = self.new_frame

            self.stack_frames = np.concatenate((self.last_frame, self.new_frame), axis=2)
            return self.transforms(self.stack_frames).to(self.device)

if __name__ =="__main__":
    var = RootVariables()
    device = torch.device("cpu")
    datasets = IMU_GAZE_DATASET(var.root, 150, device)
