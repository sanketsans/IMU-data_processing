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
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.starting_frame_index)
        self.dataframes = GET_DATAFRAME_FILES(self.rootfolder, self.total_frame_count)

        ## GAZE
        self.gaze_data_frame = self.dataframes.get_gaze_dataframe().T
        self.gaze_pts = []
        self.gaze_pts_stack = []

        ## IMU
        self.imu_data_frame = self.dataframes.get_imu_dataframe()
        self.imu_data_frame = self.imu_data_frame.iloc[self.starting_frame_index:self.total_frame_count-self.starting_frame_index+1].T
        self.imu_data_pts = []

        ## FRAMES
        self.stack_frames = []
        self.ret, self.first_frame =  self.capture.read()               ## FRAME 1
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.last_frame = cv2.resize(self.first_frame, (512, 512))
        self.new_frame = self.last_frame
        self.last_gaze_pt = self.get_gaze_pts_per_frame(self.starting_frame_index)
        self.new_gaze_pt = self.last_gaze_pt
        self.frames_included = 1
        self.frames_included = self.populate_data(self.first_frame)


    def __len__(self):
        return len(self.gaze_pts_stack)

    def get_gaze_pts_per_frame(self, frame_index):
        gaze_values = self.gaze_data_frame.iloc[:, frame_index][1:]
        self.gaze_pts = []
        # gaze_pts = []
        for index, data in enumerate(gaze_values):
            try:
                self.gaze_pts.append(tuple(ast.literal_eval(data)))
                # (gaze_x, gaze_y) = ast.literal_eval(data)
                # gaze_pt = np.array([gaze_x, gaze_y])
                # gaze_pts.append(tuple((gaze_x, gaze_y)))
            except Exception as e:
                print(e)

        return self.gaze_pts

    # def get_new_first_frame(self, capture, target_starting_frame_index):
    #     self.capture.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
    #     ret, first_frame = capture.read()
    #     curr_frame = 0
    #     for i in range(target_starting_frame_index):
    #         if ret == True:
    #             ret, new_frame = capture.read()
    #             curr_frame += 1
    #             first_frame = new_frame
    #
    #     return ret, first_frame, curr_frame
        # return self.first_frame


    def populate_data(self, first_frame):
        # for frame_num in tqdm(range(10), desc="Building frame dataset"):
        for frame_num in tqdm(range(self.starting_frame_index, self.total_frame_count-self.starting_frame_index), desc="Building frame dataset"):
            if self.ret == True:
                self.ret, self.new_frame = self.capture.read()
                self.new_frame = cv2.resize(self.new_frame, (512, 512))
                ## cv2.imwrite('frame%d.png'.format(frame_num), new_frame)
                # stack_frame = np.concatenate((last_frame, new_frame), axis=2)
                # stack_frame = self.transforms(stack_frame)
                # self.stack_frames.append(stack_frame)
                self.stack_frames.append(self.transforms(np.concatenate((self.last_frame, self.new_frame), axis=2)))

                self.last_frame = self.new_frame

                self.new_gaze_pt = self.get_gaze_pts_per_frame(self.starting_frame_index+self.frames_included)
                self.gaze_pts_stack.append(np.array([self.last_gaze_pt, self.new_gaze_pt]))
                self.frames_included += 1

                self.last_gaze_pt = self.new_gaze_pt
                self.last_gaze_pt = self.new_gaze_pt

        return self.frames_included

    def __getitem__(self, index):
        imu_values = self.imu_data_frame.iloc[:, index][1:]
        self.imu_data_pts = []
        for i, data in enumerate(imu_values):
            try:
                (acc, gyro) = ast.literal_eval(data)
                data_pt = np.array(acc + gyro)
                data_pt[1] += 9.80665
                self.imu_data_pts.append(np.round(data_pt, 3))

            except Exception as e:
                print(e)
        # print(self.gaze_pts_stack[index].shape, self.imu_data_pts.shape)
        return self.stack_frames[index].to(self.device), torch.from_numpy(self.gaze_pts_stack[index]).to(self.device), torch.from_numpy(np.array(self.imu_data_pts)).to(self.device)

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
    folder = 'imu_BookShelf_S1/'
    var = Variables()
    os.chdir(var.root + folder)
    dataset = FRAME_IMU_DATASET(var.root, folder, 800)
    print(len(dataset))
    x, y, z = dataset[5]
    a, b, c = dataset[6]
    print(z, c)

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
