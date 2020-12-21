import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import sys, ast
import numpy as np
import cv2
from tqdm import tqdm
sys.path.append('../')
from variables import Variables
from gaze_plotter import GET_DATAFRAME_FILES

class FRAME_IMU_DATASET(Dataset):
    def __init__(self, root, rootfolder, device=None, video_file='scenevideo.mp4'):
        self.root = root
        self.rootfolder = rootfolder
        self.path = self.root + self.rootfolder  + '/' if self.rootfolder[-1]!='/' else (self.root + self.rootfolder)
        self.device = device
        ## FRAMES
        self.stack_frames = []
        self.video_file = video_file
        self.starting_frame_index = 0
        self.capture = cv2.VideoCapture(self.video_file)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ret, self.first_frame = self.capture.read()         ## FRAME 1
        self.transforms = transforms.Compose([transforms.ToTensor()])

        dataframes = GET_DATAFRAME_FILES(self.frame_count)
        self.gaze_data_frame = dataframes.get_gaze_dataframe().T
        self.gaze_pts_stack = []

        ## IMU
        self.imu_data_frame = dataframes.get_imu_dataframe().T

    def __len__(self):
        return self.frame_count

    def get_gaze_pts_per_frame(self, frame_index):
        gaze_values = self.gaze_data_frame.iloc[:, frame_index][1:]
        gaze_pts = []
        for index, data in enumerate(gaze_values):
            try:
                (gaze_x, gaze_y) = ast.literal_eval(data)
                gaze_pt = np.array([gaze_x, gaze_y])
                gaze_pts.append(tuple((gaze_x, gaze_y)))
            except Exception as e:
                print(e)

        return gaze_pts

    def get_new_first_frame(self, first_frame, target_starting_frame_index):
        self.starting_frame_index = target_starting_frame_index
        for frame_num in range(target_starting_frame_index - 1):
            if self.ret == True:
                self.ret, new_frame = self.capture.read()

        self.first_frame = new_frame
        # return self.first_frame


    def populate_data(self, first_frame):
        print(self.starting_frame_index)
        last_frame = cv2.resize(first_frame, (512, 512))
        last_gaze_pt = self.get_gaze_pts_per_frame(self.starting_frame_index)
        # for frame_num in tqdm(range(self.starting_frame_index, self.frame_count-100), desc="Building frame dataset"):
        for frame_num in tqdm(range(10), desc="Building frame dataset"):
            if self.ret == True:
                self.ret, new_frame = self.capture.read()
                new_frame = cv2.resize(new_frame, (512, 512))
                # cv2.imwrite('frame%d.png'.format(frame_num), new_frame)
                stack_frame = np.concatenate((last_frame, new_frame), axis=2)
                stack_frame = self.transforms(stack_frame)
                self.stack_frames.append(stack_frame)

                last_frame = new_frame

                self.starting_frame_index += 1
                new_gaze_pt = self.get_gaze_pts_per_frame(self.starting_frame_index)
                self.gaze_pts_stack.append(np.array([last_gaze_pt, new_gaze_pt]))

                last_gaze_pt = new_gaze_pt
            else:
                break

    def __getitem__(self, index):
        imu_values = self.imu_data_frame.iloc[:, index][1:]
        imu_data_pts = []
        for index, data in enumerate(imu_values):
            try:
                (acc, gyro) = ast.literal_eval(data)
                data_pt = np.array(acc + gyro)
                data_pt[1] += 9.80665
                imu_data_pts.append(np.round(data_pt, 3))
            except Exception as e:
                print(e)

        return self.stack_frames[index].to(self.device), torch.from_numpy(self.gaze_pts_stack[index]).to(self.device), torch.from_numpy(np.array(imu_data_pts)).to(self.device)

# class IMUDataset(Dataset):
#     def __init__(self, root, rootfolder, device=None):
#         self.root = root
#         self.rootfolder = rootfolder
#         self.path = self.root + self.rootfolder  + '/' if self.rootfolder[-1]!='/' else (self.root + self.rootfolder)
#         self.imu_data_frame = pd.read_csv(self.path + 'imu_file.csv').T
#         self.device = device
#
#     def __len__(self):
#         return len(self.imu_data_frame)
#
#     def __getitem__(self, index):
#         imu_values = self.imu_data_frame.iloc[:, index][1:]
#         data_pts = []
#         for index, data in enumerate(imu_values):
#             try:
#                 (acc, gyro) = ast.literal_eval(data)
#                 data_pt = np.array(acc + gyro)
#                 data_pt[1] += 9.80665
#                 data_pts.append(np.round(data_pt, 3))
#             except Exception as e:
#                 print(e)
#
#         return np.array(data_pts)
#
# class ImageDataset(Dataset):
#     def __init__(self, root, rootfolder, device, video_file='scenevideo.mp4'):
#         self.root = root
#         self.rootfolder = rootfolder
#         self.stack_frames = []
#         self.device = device
#         self.path = self.root + self.rootfolder  + '/' if self.rootfolder[-1]!='/' else (self.root + self.rootfolder)
#         self.video_file = self.path + video_file
#         self.frame_index = 0
#
#         self.capture = cv2.VideoCapture(self.video_file)
#         self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.ret, self.first_frame = self.capture.read()         ## FRAME 1
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#
#         dataframes = GET_DATAFRAME_FILES(self.frame_count)
#         self.gaze_data_frame = dataframes.get_gaze_dataframe().T
#         self.gaze_pts_stack = []
#         # transforms.ToPILImage(),transforms.Resize((512, 512)),
#
#     def __len__(self):
#         return len(self.stack_frames)
#
#     def get_gaze_pts_per_frame(self, frame_index):
#         gaze_values = self.gaze_data_frame.iloc[:, frame_index][1:]
#         gaze_pts = []
#         for index, data in enumerate(gaze_values):
#             try:
#                 (gaze_x, gaze_y) = ast.literal_eval(data)
#                 gaze_pt = np.array([gaze_x, gaze_y])
#                 gaze_pts.append(tuple((gaze_x, gaze_y)))
#             except Exception as e:
#                 print(e)
#
#         return gaze_pts
#
#     def populate_data(self, last_frame, frame_index):
#         last_frame = cv2.resize(last_frame, (512, 512))
#         last_gaze_pt = self.get_gaze_pts_per_frame(frame_index)
#         for frame_num in tqdm(range(10), desc="Building frame dataset"):
#             if self.ret == True:
#                 self.ret, new_frame = self.capture.read()
#                 new_frame = cv2.resize(new_frame, (512, 512))
#                 # cv2.imwrite('frame%d.png'.format(frame_num), new_frame)
#                 stack_frame = np.concatenate((last_frame, new_frame), axis=2)
#                 stack_frame = self.transforms(stack_frame)
#                 self.stack_frames.append(stack_frame)
#
#                 last_frame = new_frame
#
#                 frame_index += 1
#                 new_gaze_pt = self.get_gaze_pts_per_frame(frame_index)
#                 self.gaze_pts_stack.append(np.array([last_gaze_pt, new_gaze_pt]))
#
#                 last_gaze_pt = new_gaze_pt
#             else:
#                 break
#
#     def __getitem__(self, index):
#
#         return self.stack_frames[index].to(self.device), torch.from_numpy(self.gaze_pts_stack[index]).to(self.device)

if __name__ == "__main__":

    var = Variables()
    print(var.root)
