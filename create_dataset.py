import sys, os
import numpy as np
import torch.nn as nn
import cv2
import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Dataset
from torchvision import transforms
sys.path.append('../')
# from FlowNetPytorch.models import FlowNetS
from variables import RootVariables
from helpers import standarization

class All_Dataset:
    def __init__(self):
        self.var = RootVariables()

    def get_dataset(self, folder_type, feat, labels, index):
        if index == 0:
            return self.SIG_FINAL_DATASET(feat, labels)
        elif index == 1:
            return self.VIS_FINAL_DATASET(folder_type, labels)
        else:
            return self.FusionPipeline(folder_type, feat, labels)

    class FUSION_DATASET(Dataset):
        def __init__(self, folder_type, imu_feat, labels):
            self.imu_data = []
            self.indexes = []
            self.folder_type = folder_type
            checkedLast = False
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                imu_check = np.isnan(imu_feat[index])
                if check.any() or imu_check.any():
                    continue
                else:
                    self.indexes.append(index)
                    self.imu_data.append(imu_feat[index])

            self.imu_data = standarization(self.imu_data)

            self.transforms = transforms.Compose([transforms.ToTensor()])

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
    #        img = self.frames[f_index]
            img =  np.load(self.var.root + self.folder_type + '/frames_' + str(f_index) +'.npy')
            targets = self.gaze_data[f_index]
            targets[:,0] *= 512.0
            targets[:,1] *= 384.0

            return self.transforms(img).to("cuda:0"), torch.from_numpy(self.imu_data[index]).to("cuda:0"), torch.from_numpy(targets).to("cuda:0")

    class VIS_FINAL_DATASET(Dataset):
        def __init__(self, folder_type, labels):
            self.labels = labels
            self.indexes = []
            self.folder_type = folder_type
            checkedLast = False
            for index in range(len(self.labels)):
                check = np.isnan(self.labels[index])
                if check.any():
                    continue
                else:
                    self.indexes.append(index)

            self.transforms = transforms.Compose([transforms.ToTensor()])
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            index = self.indexes[index]

            img =  np.load(self.var.root + self.folder_type + '/frames_' + str(index) +'.npy')
            targets = self.labels[index]
            #targets[:,0] *= 0.2667
            #targets[:,1] *= 0.3556

            targets[:,0] *= 512.0
            targets[:,1] *= 384.0

            return self.transforms(img).to("cuda:0"), torch.from_numpy(targets).to("cuda:0")

    class SIG_FINAL_DATASET(Dataset):
        def __init__(self, feat, labels):
            self.gaze_data, self.imu_data = [], []
            checkedLast = False
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                imu_check = np.isnan(feat[index])
                if check.any() or imu_check.any():
                    continue
                else:
                    self.gaze_data.append(labels[index])
                    self.imu_data.append(feat[index])

            self.imu_data = standarization(self.imu_data)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def __len__(self):
            return len(self.gaze_data) # len(self.labels)

        def __getitem__(self, index):
            targets = self.gaze_data[index]
            targets[:,0] *= 512.0
            targets[:,1] *= 384.0

            return torch.from_numpy(self.imu_data[index]).to(self.device), torch.from_numpy(targets).to(self.device)
