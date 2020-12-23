import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import sys, ast, os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append('../')
from variables import Variables
from gaze_plotter import GET_DATAFRAME_FILES

class FRAME_IMU_DATASET(Dataset):
    def __init__(self, root, rootfolder, target_starting_frame_index, device=None, video_file='scenevideo.mp4'):
        self.root = root
        self.rootfolder = rootfolder
        self.path = self.root + self.rootfolder  + '/' if self.rootfolder[-1]!='/' else (self.root + self.rootfolder)
        self.device = device

        self.starting_frame_index = target_starting_frame_index
        self.capture = cv2.VideoCapture(video_file)
        self.total_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stack_frames = None
        self.last_frame, self.new_frame = None, None
        self.transforms = transforms.Compose([transforms.ToTensor()])
        # self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.starting_frame_index)
        self.dataframes = GET_DATAFRAME_FILES(self.rootfolder, self.total_frame_count)

        ## GAZE
        self.gaze_data_frame = self.dataframes.get_gaze_dataframe()
        self.gaze_data_frame = self.gaze_data_frame.iloc[self.starting_frame_index:self.total_frame_count-self.starting_frame_index+1].T
        self.gaze_data_pts = []
        self.gaze_dataframe_values = None
        self.gaze_pts_stack = []

        ## IMU
        self.imu_data_frame = self.dataframes.get_imu_dataframe()
        self.imu_data_frame = self.imu_data_frame.iloc[self.starting_frame_index:self.total_frame_count-self.starting_frame_index+1].T
        self.imu_data_pts = []
        self.imu_dataframe_values = None
        self.imu_pts_stack = []


    def __len__(self):
        return len(self.gaze_data_frame.T)

    def get_imu_data_pts_from_dataframe_values(self, imu_data_pts, dataframe_values):
        for data in dataframe_values:
            (acc, gyro) = ast.literal_eval(data)
            data_pt = np.array(acc + gyro)
            data_pt[1] += 9.80665
            imu_data_pts.append(np.round(data_pt, 3))

        return imu_data_pts

    def get_gaze_data_pts_from_dataframe_values(self, gaze_data_pts, dataframe_values):
        for data in dataframe_values:
            gaze_data_pts.append(tuple(ast.literal_eval(data)))

        return gaze_data_pts

    def __getitem__(self, index):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.starting_frame_index+ index)
        self.gaze_pts_stack, self.imu_pts_stack = [], []
        for i in range(2):
            self.imu_dataframe_values = self.imu_data_frame.iloc[:, index+i][1:]
            self.gaze_dataframe_values = self.gaze_data_frame.iloc[:, index+i][1:]
            self.imu_data_pts, self.gaze_data_pts = [], []
            self.imu_data_pts = self.get_imu_data_pts_from_dataframe_values(self.imu_data_pts, self.imu_dataframe_values)
            self.gaze_data_pts = self.get_gaze_data_pts_from_dataframe_values(self.gaze_data_pts, self.gaze_dataframe_values)

            self.gaze_pts_stack.append(self.gaze_data_pts)
            self.imu_pts_stack.append(self.imu_data_pts)


            self.ret, self.new_frame = self.capture.read()
            self.new_frame = cv2.resize(self.new_frame, (512, 512))
            if i == 0:
                self.last_frame = self.new_frame

        self.stack_frames = np.concatenate((self.last_frame, self.new_frame), axis=2)

        return self.transforms(self.stack_frames).to(self.device), torch.tensor(self.gaze_pts_stack).to(self.device), torch.tensor(self.imu_pts_stack).to(self.device)

if __name__ == "__main__":
    folder = 'imu_BookShelf_S1/'
    var = Variables()
    os.chdir(var.root + folder)
    dataset = FRAME_IMU_DATASET(var.root, folder, 600)
    print(len(dataset))
    x, y, z = dataset[5]
    a, b, c = dataset[6]
    print(y, z)
    # print(b, c)

    # video_file = 'scenevideo.mp4'
    # capture = cv2.VideoCapture(video_file)
    # total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # capture.set(cv2.CAP_PROP_POS_FRAMES,500)
    # while  True:
    #     ret, frame = capture.read()
    #     cv2.imshow('image', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # plt.show()
    # dataframes = GET_DATAFRAME_FILES(total_frame_count)
    # gaze_data_frame = dataframes.get_gaze_dataframe()
    # gaze_data_frame = gaze_data_frame.iloc[0:5].T
    # print(gaze_data_frame)
    # print(gaze_data_frame.iloc[:,0])
    # print(len(dataset))
