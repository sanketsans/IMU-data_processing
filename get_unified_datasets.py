import os, json, ast
import matplotlib.pyplot as plt
import sys, math
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
sys.path.append('../')
from proc import BUILDING_DATASET
from variables import RootVariables

class DATAFRAMES_AND_UNIFIED_DATASETS:
    def __init__(self, trim_frame_size):
        self.var = RootVariables()
        self.trim_size = trim_frame_size
        self.dataset = None
        self.dataset = None
        self.imu_arr_acc, self.imu_arr_gyro, self.gaze_array = None, None, None
        self.last = None
        self.new = None
        self.temp = None

        self.panda_data = {}

    def load_unified_gaze_dataset(self):        ## missing data in imu_lift_s1
        for index, subDir in enumerate(tqdm(os.listdir(self.var.root), desc="Building gaze dataset")):
            if 'imu_' in subDir:
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(self.var.root + subDir)
                video_file = 'scenevideo.mp4'
                capture = cv2.VideoCapture(video_file)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                self.dataset = BUILDING_DATASET(subDir)
                self.dataset.POP_GAZE_DATA(frame_count)
                self.gaze_arr = np.array(self.dataset.var.gaze_data).transpose()
                # print(gaze_arr)
                self.temp = np.zeros((frame_count*4-self.trim_size*4, 2))
                self.temp[:,0] = self.gaze_arr[tuple([np.arange(self.trim_size*4, frame_count*4), [0]])]
                self.temp[:,1] = self.gaze_arr[tuple([np.arange(self.trim_size*4, frame_count*4), [1]])]

                if index > 0:
                    self.new = np.concatenate((self.last, self.temp), axis=0)

                self.last = self.temp

        return self.new

    def load_unified_imu_dataset(self):     ## missing data in imu_CoffeeVendingMachine_S2
        for index, subDir in enumerate(tqdm(os.listdir(self.var.root), desc="Building imu dataset")):
            if 'imu_' in subDir:
                # print(subDir)
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(self.var.root + subDir)
                video_file = 'scenevideo.mp4'
                capture = cv2.VideoCapture(video_file)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                self.dataset = BUILDING_DATASET(subDir)
                self.dataset.POP_IMU_DATA(frame_count)
                self.imu_arr_acc = np.array(self.dataset.var.imu_data_acc).transpose()
                self.imu_arr_gyro = np.array(self.dataset.var.imu_data_gyro).transpose()
                # print(gaze_arr)
                self.temp = np.zeros((frame_count*4-self.trim_size*4, 6))
                self.temp[:,0] = self.imu_arr_acc[tuple([np.arange(self.trim_size*4, frame_count*4), [0]])]
                self.temp[:,1] = self.imu_arr_acc[tuple([np.arange(self.trim_size*4, frame_count*4), [1]])]
                self.temp[:,2] = self.imu_arr_acc[tuple([np.arange(self.trim_size*4, frame_count*4), [2]])]
                self.temp[:,3] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*4, frame_count*4), [0]])]
                self.temp[:,4] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*4, frame_count*4), [1]])]
                self.temp[:,5] = self.imu_arr_gyro[tuple([np.arange(self.trim_size*4, frame_count*4), [2]])]

                if index > 0:
                    self.new = np.concatenate((self.last, self.temp), axis=0)

                self.last = self.temp

        return self.new

    def create_dataframes(self, subDir, start_index=0):
        subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
        os.chdir(self.var.root + subDir)
        video_file = 'scenevideo.mp4'
        capture = cv2.VideoCapture(video_file)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.dataset = BUILDING_DATASET(subDir)
        self.dataset.POP_GAZE_DATA(frame_count)
        self.dataset.POP_IMU_DATA(frame_count)
        ## GAZE
        for sec in range(frame_count):
            self.panda_data[sec] = list(zip(self.dataset.var.gaze_data[0][start_index:start_index + 4], self.dataset.var.gaze_data[1][start_index:start_index+4]))
            start_index += 4

        self.df_gaze = pd.DataFrame({ key:pd.Series(value) for key, value in self.panda_data.items()}).T
        self.df_gaze.columns =['Gaze_Pt_1', 'Gaze_Pt_2', 'Gaze_Pt_3', 'Gaze_Pt_4']
        self.df_gaze.to_csv('gaze_file.csv')
        # self.df_gaze = pd.read_csv('gaze_file.csv')

        ## IMU
        start_index = 0
        for sec in range(frame_count):
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
    dataframes = GET_DATAFRAME_FILES(100)
    gaze_datas = dataframes.load_unified_gaze_dataset()
    imu_datas = dataframes.load_unified_imu_dataset()
    
