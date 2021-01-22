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
from helpers import Helpers

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
        checkedLast = False
        while True:
            check = np.isnan(self.gaze_data[index])
            if check.any():
                index = (index - 1) if checkedLast else (index + 1)
                if index == self.__len__():
                    checkedLast = True
            else:
                break
        return self.transforms(self.frame_data[index]).to(self.device), torch.from_numpy(np.concatenate((self.imu_data[index], self.imu_data[index+1]), axis=0)).to(self.device), torch.from_numpy(self.gaze_data[index]*1000.0).to(self.device)

class IMU_GAZE_FRAME_DATASET:
    def __init__(self, root, frame_size, trim_size, distribution='N'):
        self.root = root
        self.dataset = BUILDING_DATASETS(self.root, frame_size, trim_size)
        self.frame_datasets = None
        self.imu_train_datasets, self.gaze_train_datasets = None, None
        self.imu_test_datasets, self.gaze_test_datasets = None, None
        if Path(self.root + 'imuExtracted_training_data_' + str(trim_size) + '.npy').is_file():
            print('Files exists')
            self.imu_train_datasets = np.load(self.root + 'imuExtracted_training_data_' + str(trim_size) + '.npy')
            self.gaze_train_datasets = np.load(self.root + 'gazeExtracted_training_data_' + str(trim_size) + '.npy')
            self.imu_test_datasets = np.load(self.root + 'imuExtracted_testing_data_' + str(trim_size) + '.npy')
            self.gaze_test_datasets = np.load(self.root + 'gazeExtracted_testing_data_' + str(trim_size) + '.npy')
        else:
            print('saved files does not exis')
            self.imu_train_datasets, self.imu_test_datasets = self.dataset.load_unified_imu_dataset()
            self.gaze_train_datasets, self.gaze_test_datasets = self.dataset.load_unified_gaze_dataset()
            np.save(self.root + 'imuExtracted_training_data_' + str(trim_size) + '.npy', self.imu_train_datasets)
            np.save(self.root + 'gazeExtracted_training_data_' + str(trim_size) + '.npy', self.gaze_train_datasets)
            np.save(self.root + 'imuExtracted_testing_data_' + str(trim_size) + '.npy', self.imu_test_datasets)
            np.save(self.root + 'gazeExtracted_testing_data_' + str(trim_size) + '.npy', self.gaze_test_datasets)

        self.frame_datasets = self.dataset.load_unified_frame_dataset()

        if distribution == 'N':
            self.imu_train_datasets = self.dataset.normalization(self.imu_train_datasets)
            self.imu_test_datasets = self.dataset.normalization(self.imu_test_datasets)
        else:
            self.imu_train_datasets = self.dataset.standarization(self.imu_train_datasets)
            self.imu_test_datasets = self.dataset.standarization(self.imu_test_datasets)

        self.gaze_train_datasets = self.gaze_train_datasets.reshape(-1, 4, self.gaze_train_datasets.shape[-1])
        self.imu_train_datasets = self.imu_train_datasets.reshape(-1, 4, self.imu_train_datasets.shape[-1])

        self.gaze_test_datasets = self.gaze_test_datasets.reshape(-1, 4, self.gaze_test_datasets.shape[-1])
        self.imu_test_datasets = self.imu_test_datasets.reshape(-1, 4, self.imu_test_datasets.shape[-1])

    def __len__(self):
        return int(len(self.imu_datasets))      ## number of frames corresponding to

    # def __getitem__(self, index):
    #     return self.imu_datasets[index], self.gaze_datasets[index]

if __name__ =="__main__":
    var = RootVariables()
    device = torch.device("cpu")
    trim_size = 150
    frame_size = 256
    datasets = IMU_GAZE_FRAME_DATASET(var.root, frame_size, trim_size)
    train_imu_dataset = datasets.imu_train_datasets
    test_imu_dataset = datasets.imu_test_datasets

    train_gaze_dataset = datasets.gaze_train_datasets
    test_gaze_dataset = datasets.gaze_test_datasets
    print(train_imu_dataset[0], test_imu_dataset[0])
    # folders_num, gaze_start_index, gaze_end_index, trim_size = 0, 0, 0, 150
    # imu_start_index, imu_end_index = 0, 0
    # utls = Helpers()
    # sliced_imu_dataset, sliced_gaze_dataset = None, None
    # for index, subDir in enumerate(sorted(os.listdir(var.root))):
    #     if 'imu_' in subDir:
    #         folders_num += 1
    #         print(subDir)
    #         subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
    #         os.chdir(var.root + subDir)
    #         capture = cv2.VideoCapture('scenevideo.mp4')
    #         frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #         gaze_end_index = gaze_start_index + frame_count - trim_size*2
    #         imu_end_index = imu_start_index + frame_count - trim_size
    #         sliced_imu_dataset = uni_imu_dataset[imu_start_index: imu_end_index]
    #         sliced_gaze_dataset = uni_gaze_dataset[gaze_start_index: gaze_end_index]
    #         dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
    #         print(len(dataset))
    #         i, g = dataset[1]
    #         print(g/1000.0)
    #         print(i.shape, g.shape)
    #         print(i[0], i[-1])
    #
    #         gaze_start_index = gaze_end_index
    #         imu_start_index = imu_end_index
    #     if 'imu_CoffeeVendingMachine_S2' in subDir :
    #         break
    #
    # print(sliced_imu_dataset[0].shape)
    # print('\n')
