import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import sys, ast
import numpy as np
import cv2
sys.path.append('../')
from variables import Variables

class IMUDataset(Dataset):
    def __init__(self, root, rootfolder, device=None):
        self.root = root
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

        return (np.array(data_pts))

class ImageDataset(Dataset):
    def __init__(self, root, rootfolder, video_file='scenevideo.mp4', device=None):
        self.root = root
        self.rootfolder = rootfolder
        self.path = self.root + self.rootfolder  + '/' if self.rootfolder[-1]!='/' else (self.root + self.rootfolder)
        self.video_file = self.path + video_file

        self.capture = cv2.VideoCapture(self.video_file)
        self.ret, self.frame = self.capture.read()         ## FRAME 1
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if self.ret == True:
            ret, frame2 = self.capture.read()

        stack_frame = np.concatenate((self.frame, frame2), axis=2)
        stack_frame = self.transforms(stack_frame)
        # print(stack_frame.shape)

        return stack_frame

if __name__ == "__main__":

    var = Variables()
    print(var.root)
