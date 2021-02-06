import os
import sys, math
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
sys.path.append('../')
from loader import JSON_LOADER
from variables import RootVariables
import matplotlib.pyplot as plt

class BUILDING_DATASETS:
    def __init__(self, root, frame_size, trim_frame_size):
        self.root = root
        self.frame_size = frame_size
        self.trim_size = trim_frame_size
        self.dataset = None
        self.imu_arr_acc, self.imu_arr_gyro, self.gaze_arr = None, None, None
        self.last = None
        self.train_new, self.test_new = None, None
        temp = None
        self.video_file = 'scenevideo.mp4'
        self.folders_num = 0
        self.frame_count = 0
        self.capture = None
        self.ret = None
        self.toggle = 0
        self.stack_frames = []

        self.panda_data = {}

    def populate_gaze_data(self, subDir, toggle=1):
        if toggle != self.toggle:
            self.folders_num = 0
            self.toggle = toggle

        subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
        print(subDir)
        os.chdir(self.root + subDir)
        capture = cv2.VideoCapture(self.video_file)
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.dataset = JSON_LOADER(subDir)
        self.dataset.POP_GAZE_DATA(self.frame_count)
        self.gaze_arr = np.array(self.dataset.var.gaze_data).transpose()
        _ = os.system('rm gaze_file.csv')
        self.panda_data = {}
        self.create_dataframes(subDir, 'gaze')
        self.gaze_arr = np.array(self.dataset.var.gaze_data).transpose()
#         temp = np.zeros((len(self.gaze_arr) , 6))
# #         print(len(self.imu_arr_acc), len(temp))
#         temp[:,0] = self.gaze_arr[:, 0]
#         temp[:,1] = self.gaze_arr[:,1]

        temp = np.zeros((self.frame_count*4-self.trim_size*4*2, 2))
        temp[:,0] = self.gaze_arr[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [0]])]
        temp[:,1] = self.gaze_arr[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [1]])]

        return temp

    def load_unified_gaze_dataset(self):        ## missing data in imu_lift_s1
        self.folders_num = 0
        for index, subDir in enumerate(tqdm(sorted(os.listdir(self.root)), desc="Building gaze dataset")):
            if 'train_' in subDir :
                self.temp = self.populate_gaze_data(subDir, 1)
                self.folders_num += 1
                if self.folders_num > 1:
                    self.train_new = np.concatenate((self.last, self.temp), axis=0)
                else:
                    self.train_new = self.temp
                self.last = self.train_new
            if 'test_' in subDir:
                self.temp = self.populate_gaze_data(subDir, -1)
                self.folders_num += 1
                if self.folders_num > 1:
                    self.test_new = np.concatenate((self.last, self.temp), axis=0)
                else:
                    self.test_new = self.temp
                self.last = self.test_new

        return self.train_new, self.test_new

    def load_unified_frame_dataset(self):
        ## INCLUDES THE LAST FRAME
        self.folders_num = 0
        for index, subDir in enumerate(tqdm(sorted(os.listdir(self.root)), desc="Building Image Dataset")):
            if 'train_' in subDir or 'val_' in subDir or 'test_' in subDir:
                total_frames = 0
                self.folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(self.root + subDir)
                self.capture = cv2.VideoCapture(self.video_file)
                self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if not Path(str(self.frame_size) + '_framesExtracted_data_' + str(self.trim_size) + '.npy').is_file():
                    print(subDir)
                    # _ = os.system('rm framesExtracted_data_' + str(self.trim_size) + '.npy')
                    os.chdir(self.root + subDir)
                    self.capture = cv2.VideoCapture(self.video_file)
                    self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

                    self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.trim_size)
                    self.ret, self.last = self.capture.read()
                    self.last = cv2.cvtColor(self.last, cv2.COLOR_BGR2RGB)
                    self.last = cv2.resize(self.last, (self.frame_size, self.frame_size))
                    total_frames = 1
                    while self.ret:
                        if total_frames == (self.frame_count - self.trim_size*2):
                            break
                        self.ret, self.train_new = self.capture.read()
                        self.train_new = cv2.cvtColor(self.train_new, cv2.COLOR_BGR2RGB)
                        self.train_new = cv2.resize(self.train_new, (self.frame_size, self.frame_size))
                        total_frames += 1
                        self.stack_frames.append(np.concatenate((self.last, self.train_new), axis=2))
                        self.last = self.train_new

                    with open(self.root + subDir + str(self.frame_size) + '_framesExtracted_data_' + str(self.trim_size) + '.npy', 'wb') as f:
                        np.save(f, self.stack_frames)
                        f.close()

                    self.stack_frames = []

        return self.train_new

    def populate_imu_data(self, subDir, toggle=1):
        if toggle != self.toggle:
            self.folders_num = 0
            self.toggle = toggle

        subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
        print(subDir)
        os.chdir(self.root + subDir)
        capture = cv2.VideoCapture(self.video_file)
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.dataset = JSON_LOADER(subDir)
        self.dataset.POP_IMU_DATA(self.frame_count, cut_short=False)
        _ = os.system('rm imu_file.csv')
        self.panda_data = {}
        self.create_dataframes(subDir, dframe_type='imu')

        # _ = os.system('rm folder_imuExtracted_data_' + str(self.trim_size) + '.pt')
        self.imu_arr_acc = np.array(self.dataset.var.imu_data_acc).transpose()
        self.imu_arr_gyro = np.array(self.dataset.var.imu_data_gyro).transpose()
        temp = np.zeros((len(self.imu_arr_acc) , 6))
#         print(len(self.imu_arr_acc), len(temp))
#         temp[:,0] = self.imu_arr_acc[:, 0]
#         temp[:,1] = self.imu_arr_acc[:,1]
#         temp[:,2] = self.imu_arr_acc[:,2]
#         temp[:,3] = self.imu_arr_gyro[:,0]
#         temp[:,4] = self.imu_arr_gyro[:,1]
#         temp[:,5] = self.imu_arr_gyro[:,2]
#         temp = np.zeros((self.frame_count*4 - self.trim_size*4*2 + 4*24, 6))
#         temp[:,0] = self.imu_arr_acc[tuple([np.arange(self.trim_size*4 - 4*9, self.frame_count*4 - self.trim_size*4 + 4*15), [0]])]
        # temp = np.zeros((self.frame_count*4 - self.trim_size*4, 6))
        # temp[:,0] = self.imu_arr_acc[tuple([np.arange(self.trim_size*2, self.frame_count*4 - self.trim_size*2), [0]])]
        # temp[:,0] = self.imu_arr_acc[tuple([np.arange(self.trim_size*2 , self.frame_count*4 - self.trim_size*2 ), [0]])]
        # temp[:,1] = self.imu_arr_acc[tuple([np.arange(self.trim_size*2 , self.frame_count*4- self.trim_size*2 ), [1]])]
        # temp[:,2] = self.imu_arr_acc[tuple([np.arange(self.trim_size*2 , self.frame_count*4- self.trim_size*2 ), [2]])]
        # temp[:,3] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*2 , self.frame_count*4- self.trim_size*2 ), [0]])]
        # temp[:,4] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*2 , self.frame_count*4- self.trim_size*2 ), [1]])]
        # temp[:,5] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*2 , self.frame_count*4- self.trim_size*2 ), [2]])]
        temp = np.zeros((self.frame_count*4-self.trim_size*4, 6))
        temp[:,0] = self.imu_arr_acc[tuple([np.arange(self.trim_size*2, self.frame_count*4 - self.trim_size*2), [0]])]
        temp[:,1] = self.imu_arr_acc[tuple([np.arange(self.trim_size*2, self.frame_count*4 - self.trim_size*2), [1]])]
        temp[:,2] = self.imu_arr_acc[tuple([np.arange(self.trim_size*2, self.frame_count*4 - self.trim_size*2), [2]])]
        temp[:,3] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*2, self.frame_count*4 - self.trim_size*2), [0]])]
        temp[:,4] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*2, self.frame_count*4 - self.trim_size*2), [1]])]
        temp[:,5] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*2, self.frame_count*4 - self.trim_size*2), [2]])]

        return temp

    def load_unified_imu_dataset(self):     ## missing data in imu_CoffeeVendingMachine_S2
        for index, subDir in enumerate(tqdm(sorted(os.listdir(self.root)), desc="Building IMU dataset")):
            if 'train_' in subDir :
                self.temp = self.populate_imu_data(subDir, 1)
                self.folders_num += 1
                if self.folders_num > 1:
                    self.train_new = np.concatenate((self.last, self.temp), axis=0)
                else:
                    self.train_new = self.temp
                self.last = self.train_new
            if 'test_' in subDir:
                self.temp = self.populate_imu_data(subDir, -1)
                self.folders_num += 1
                if self.folders_num > 1:
                    self.test_new = np.concatenate((self.last, self.temp), axis=0)
                else:
                    self.test_new = self.temp
                self.last = self.test_new

        return self.train_new, self.test_new

    def create_dataframes(self, subDir, dframe_type, start_index=0):
        if dframe_type == 'gaze':
            ## GAZE
            for sec in range(self.frame_count):
                self.panda_data[sec] = list(zip(self.dataset.var.gaze_data[0][start_index:start_index + 4], self.dataset.var.gaze_data[1][start_index:start_index+4]))
                start_index += 4

            self.df_gaze = pd.DataFrame({ key:pd.Series(value) for key, value in self.panda_data.items()}).T
            self.df_gaze.columns =['Gaze_Pt_1', 'Gaze_Pt_2', 'Gaze_Pt_3', 'Gaze_Pt_4']
            self.df_gaze.to_csv('gaze_file.csv')

        elif dframe_type == 'imu':
            ## IMU
            for sec in range(self.frame_count):
                # self.panda_data[sec] = list(tuple((sec, sec+2)))
                self.panda_data[sec] = list(zip(zip(self.dataset.var.imu_data_acc[0][start_index:start_index+4],
                                            self.dataset.var.imu_data_acc[1][start_index:start_index+4],
                                            self.dataset.var.imu_data_acc[2][start_index:start_index+4]),

                                        zip(self.dataset.var.imu_data_gyro[0][start_index:start_index+4],
                                                self.dataset.var.imu_data_gyro[1][start_index:start_index+4],
                                                self.dataset.var.imu_data_gyro[2][start_index:start_index+4])))
                start_index += 4
            self.df_imu = pd.DataFrame({ key:pd.Series(value) for key, value in self.panda_data.items()}).T
            self.df_imu.columns =['IMU_Acc/Gyro_Pt_1', 'IMU_Acc/Gyro_Pt_2', 'IMU_Acc/Gyro_Pt_3', 'IMU_Acc/Gyro_Pt_4']
            self.df_imu.to_csv('imu_file.csv')

if __name__ == "__main__":
    var = RootVariables()
    # dataset_folder = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'
    # os.chdir(dataset_folder)
    dataframes = BUILDING_DATASETS(var.root, 256, 150)

    gaze_datas = dataframes.load_unified_gaze_dataset()
    imu_datas = dataframes.load_unified_imu_dataset()
    print(len(imu_datas))
    # imu_datas= dataframes.load_unified_imu_dataset()
    # plt.subplot(221)
    # _ = plt.hist(imu_datas[:,0], bins='auto', label='before N')
    # normal = dataframes.normalization(imu_datas)
    # _ = plt.hist(normal[:,0], bins='auto', label='after N')
    # plt.legend()

    # imu_datas= dataframes.load_unified_imu_dataset()
    # plt.subplot(222)
    # _ = plt.hist(imu_datas[:,0], bins='auto', label='before S')
    # normal = dataframes.standarization(imu_datas)
    # _ = plt.hist(normal[:,0], bins='auto', label='after S')
    # plt.legend()
    # plt.show()
