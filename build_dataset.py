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
    def __init__(self, root, trim_frame_size):
        self.root = root
        self.trim_size = trim_frame_size
        self.dataset = None
        self.imu_arr_acc, self.imu_arr_gyro, self.gaze_array = None, None, None
        self.last = None
        self.new = None
        self.temp = None
        self.video_file = 'scenevideo.mp4'
        self.folders_num = 0
        self.frame_count = 0
        self.capture = None
        self.ret = None
        self.stack_frames = []

        self.panda_data = {}

    def standarization(self, datas):
        datas = datas.reshape(-1, datas.shape[-1])
        rows, cols = datas.shape
        for i in range(cols):
            mean = torch.mean(datas[:,i])
            std = torch.std(datas[:,i])
            datas[:,i] = (datas[:,i] - mean) / std

        return datas

    def normalization(self, datas):
        datas = datas.reshape(-1, datas.shape[-1])
        rows, cols = datas.shape
        for i in range(cols):
            max = torch.max(datas[:,i])
            min = torch.min(datas[:,i])
            datas[:,i] = (datas[:,i] - min ) / (max - min)
        return datas

    def load_unified_gaze_dataset(self):        ## missing data in imu_lift_s1
        self.folders_num = 0
        for index, subDir in enumerate(tqdm(sorted(os.listdir(self.root)), desc="Building gaze dataset")):
            if 'imu_' in subDir or 'val_' in subDir or 'test_' in subDir:
                self.folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(self.root + subDir)
                capture = cv2.VideoCapture(self.video_file)
                self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                self.dataset = JSON_LOADER(subDir)
                self.dataset.POP_GAZE_DATA(self.frame_count)
                self.gaze_arr = np.array(self.dataset.var.gaze_data).transpose()
                if not Path('gaze_file.csv').is_file():
                    # print('new file wil')
                    # _ = os.system('rm gaze_file.csv')
                    # _ = os.system('rm imu_file.csv')
                    self.create_dataframes(subDir, 'gaze')

                if not Path(self.root + 'gazeExtracted_data_' + str(self.trim_size) + '.pt').is_file():
                    print(subDir)
                    # _ = os.system('rm folder_gazeExtracted_data_' + str(self.trim_size) + '.pt')
                    self.gaze_arr = np.array(self.dataset.var.gaze_data).transpose()
                    # print(gaze_arr)
                    self.temp = np.zeros((self.frame_count*4-self.trim_size*4*2, 2))
                    self.temp[:,0] = self.gaze_arr[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [0]])]
                    self.temp[:,1] = self.gaze_arr[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [1]])]


                    if self.folders_num > 1:
                        self.new = np.concatenate((self.last, self.temp), axis=0)
                    else:
                        self.new = self.temp
                    self.last = self.new
                    # torch.save(torch.from_numpy(self.temp), 'folder_gazeExtracted_data_' + str(self.trim_size) + '.pt')

        return self.new

    def load_unified_frame_dataset(self):
        self.folders_num = 0
        for index, subDir in enumerate(tqdm(sorted(os.listdir(self.root)), desc="Building Image Dataset")):
            if 'imu_' in subDir or 'val_' in subDir or 'test_' in subDir:
                total_frames = 0
                self.folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(self.root + subDir)
                self.capture = cv2.VideoCapture(self.video_file)
                self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if not Path('framesExtracted_data_' + str(self.trim_size) + '.npy').is_file():
                    print(subDir)
                    # _ = os.system('rm folder_imuExtracted_data_' + str(self.trim_size) + '.npy')
                    os.chdir(self.root + subDir)
                    self.capture = cv2.VideoCapture(self.video_file)
                    self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

                    self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.trim_size)
                    self.ret, self.last = self.capture.read()
                    self.last = cv2.cvtColor(self.last, cv2.COLOR_BGR2RGB)
                    self.last = cv2.resize(self.last, (512, 512))
                    total_frames = 1
                    while self.ret:
                        if total_frames == (self.frame_count - self.trim_size*2):
                            break
                        # cv2.imwrite(root + subDir + "frames/frame%d.jpg" % total_frames, frame)
                        self.ret, self.new = self.capture.read()
                        self.new = cv2.cvtColor(self.new, cv2.COLOR_BGR2RGB)
                        self.new = cv2.resize(self.new, (512, 512))
                        total_frames += 1


                        self.stack_frames.append(np.concatenate((self.last, self.new), axis=2))
                        # self.stack_frames.append((torch.cat((self.last, self.new), axis=0)).detach().cpu().numpy())
                        self.last = self.new
                    break

                    with open(self.root + subDir + 'framesExtracted_data_' + str(self.trim_size) + '.npy', 'wb') as f:
                        np.save(f, self.stack_frames)
                        f.close()

                    self.stack_frames = []

        return self.new

    def load_unified_imu_dataset(self):     ## missing data in imu_CoffeeVendingMachine_S2
        self.folders_num = 0
        # sum = 0
        for index, subDir in enumerate(tqdm(sorted(os.listdir(self.root)), desc="Building IMU Dataset")):
            if 'imu_' in subDir or 'val_' in subDir or 'test_' in subDir:
                self.folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(self.root + subDir)
                capture = cv2.VideoCapture(self.video_file)
                self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                self.dataset = JSON_LOADER(subDir)
                self.dataset.POP_IMU_DATA(self.frame_count)
                if not Path('imu_file.csv').is_file():
                    # print('new file wil')
                    # _ = os.system('rm imu_file.csv')
                    # _ = os.system('rm gaze_file.csv')
                    self.create_dataframes(subDir, dframe_type='imu')

                if  not Path(self.root + 'imuExtracted_data_' + str(self.trim_size) + '.pt').is_file():
                    print(subDir)
                    # _ = os.system('rm folder_imuExtracted_data_' + str(self.trim_size) + '.pt')
                    self.imu_arr_acc = np.array(self.dataset.var.imu_data_acc).transpose()
                    self.imu_arr_gyro = np.array(self.dataset.var.imu_data_gyro).transpose()
                    # print(gaze_arr)
                    self.temp = np.zeros((self.frame_count*4-self.trim_size*4*2, 6))
                    self.temp[:,0] = self.imu_arr_acc[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [0]])]
                    self.temp[:,1] = self.imu_arr_acc[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [1]])]
                    self.temp[:,2] = self.imu_arr_acc[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [2]])]
                    self.temp[:,3] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [0]])]
                    self.temp[:,4] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [1]])]
                    self.temp[:,5] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*4, self.frame_count*4 - self.trim_size*4), [2]])]


                    if self.folders_num > 1:
                        self.new = np.concatenate((self.last, self.temp), axis=0)
                    else:
                        self.new = self.temp

                    self.last = self.new
                    # sum += self.frame_count - self.trim_size*2
                    # print(self.frame_count, sum, self.temp.shape)
                    # torch.save(torch.from_numpy(self.temp), 'folder_imuExtracted_data_' + str(self.trim_size) + '.pt')

        return self.new

    def create_dataframes(self, subDir, dframe_type, start_index=0):
        if dframe_type == 'gaze':
            ## GAZE
            for sec in range(self.frame_count):
                self.panda_data[sec] = list(zip(self.dataset.var.gaze_data[0][start_index:start_index + 4], self.dataset.var.gaze_data[1][start_index:start_index+4]))
                start_index += 4

            self.df_gaze = pd.DataFrame({ key:pd.Series(value) for key, value in self.panda_data.items()}).T
            self.df_gaze.columns =['Gaze_Pt_1', 'Gaze_Pt_2', 'Gaze_Pt_3', 'Gaze_Pt_4']
            self.df_gaze.to_csv('gaze_file.csv')
            # self.df_gaze = pd.read_csv('gaze_file.csv')
        else:
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
            # self.df_imu = pd.read_csv('imu_file.csv')


if __name__ == "__main__":
    var = RootVariables()
    # dataset_folder = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'
    # os.chdir(dataset_folder)
    dataframes = BUILDING_DATASETS(var.root, 150)

    gaze_datas = dataframes.load_unified_gaze_dataset(type='imu_')
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
