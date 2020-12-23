import os, json, ast
import matplotlib.pyplot as plt
import sys, math
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import seaborn as sns
sys.path.append('../')
from proc import BUILDING_DATASET

class GET_DATAFRAME_FILES:
    def __init__(self, folder, frame_count):
        self.dataset = BUILDING_DATASET(folder)
        if Path('gaze_file.csv').is_file():
            print('File exists')
            self.df_gaze = pd.read_csv('gaze_file.csv')
            self.df_imu = pd.read_csv('imu_file.csv')

        else:
            print('File did not exists')
            self.dataset = BUILDING_DATASET(folder)
            self.panda_data = {}
            start_index = 0
            self.dataset.POP_GAZE_DATA()
            for sec in range(frame_count):
                self.panda_data[sec] = list(zip(self.dataset.var.gaze_data[0][start_index:start_index + 4], self.dataset.var.gaze_data[1][start_index:start_index+4]))
                start_index += 4

            self.df_gaze = pd.DataFrame({ key:pd.Series(value) for key, value in self.panda_data.items()}).T
            self.df_gaze.columns =['Gaze_Pt_1', 'Gaze_Pt_2', 'Gaze_Pt_3', 'Gaze_Pt_4']
            self.df_gaze.to_csv('gaze_file.csv')
            self.df_gaze = pd.read_csv('gaze_file.csv')

            ## GAZE DATAFRAME
            start_index = 0
            self.dataset.POP_IMU_DATA()
            self.dataset = self.get_normalized_values(self.dataset)
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
            # self.df_imu[['Acc_Pt_1', 'Gyro_Pt_1']] = pd.DataFrame(self.df_imu['IMU_Acc/Gyro_Pt_1'].tolist(), index=self.df_imu.index)
            # self.df_imu[['Acc_Pt_2', 'Gyro_Pt_2']] = pd.DataFrame(self.df_imu['IMU_Acc/Gyro_Pt_2'].tolist(), index=self.df_imu.index)
            # self.df_imu[['Acc_Pt_3', 'Gyro_Pt_3']] = pd.DataFrame(self.df_imu['IMU_Acc/Gyro_Pt_3'].tolist(), index=self.df_imu.index)
            # self.df_imu[['Acc_Pt_4', 'Gyro_Pt_4']] = pd.DataFrame(self.df_imu['IMU_Acc/Gyro_Pt_4'].tolist(), index=self.df_imu.index)
            # del self.df_imu['IMU_Acc/Gyro_Pt_1']
            # del self.df_imu['IMU_Acc/Gyro_Pt_2']
            # del self.df_imu['IMU_Acc/Gyro_Pt_3']
            # del self.df_imu['IMU_Acc/Gyro_Pt_4']
            self.df_imu.to_csv('imu_file.csv')
            self.df_imu = pd.read_csv('imu_file.csv')

    def get_normalized_values(self, dataset):
        dataset.var.imu_data_acc[0] = (dataset.var.imu_data_acc[0] - np.min(dataset.var.imu_data_acc[0])) / (np.max(dataset.var.imu_data_acc[0]) - np.min(dataset.var.imu_data_acc[0]))
        dataset.var.imu_data_acc[1] = (dataset.var.imu_data_acc[1] - np.min(dataset.var.imu_data_acc[1])) / (np.max(dataset.var.imu_data_acc[1]) - np.min(dataset.var.imu_data_acc[1]))
        dataset.var.imu_data_acc[2] = (dataset.var.imu_data_acc[2] - np.min(dataset.var.imu_data_acc[2])) / (np.max(dataset.var.imu_data_acc[2]) - np.min(dataset.var.imu_data_acc[2]))
        dataset.var.imu_data_gyro[0] = (dataset.var.imu_data_gyro[0] - np.min(dataset.var.imu_data_gyro[0])) / (np.max(dataset.var.imu_data_gyro[0]) - np.min(dataset.var.imu_data_gyro[0]))
        dataset.var.imu_data_gyro[1] = (dataset.var.imu_data_gyro[1] - np.min(dataset.var.imu_data_gyro[1])) / (np.max(dataset.var.imu_data_gyro[1]) - np.min(dataset.var.imu_data_gyro[1]))
        dataset.var.imu_data_gyro[2] = (dataset.var.imu_data_gyro[2] - np.min(dataset.var.imu_data_gyro[2])) / (np.max(dataset.var.imu_data_gyro[2]) - np.min(dataset.var.imu_data_gyro[2]))

        return dataset

    def get_imu_dataframe(self):
        return self.df_imu

    def get_gaze_dataframe(self):
        return self.df_gaze


if __name__ == "__main__":
    folder = sys.argv[1]
    dataset_folder = '/home/sans/Downloads/gaze_data/'
    # dataset_folder = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'
    # os.chdir(dataset_folder)
    os.chdir(dataset_folder + folder + '/' if folder[-1]!='/' else (dataset_folder + folder))

    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(frame_count, fps)
    ret, frame = capture.read()
    print(frame.shape)

    dataframes = GET_DATAFRAME_FILES(folder, frame_count)
    df_gaze = dataframes.get_gaze_dataframe()
    df_imu = dataframes.get_imu_dataframe()

    print(len(df_gaze), len(df_imu))
    # df_gaze = df_gaze.T

    # count = 0
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
    # for i in range(length):
    #     if ret == True:
    #         # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #         # cv2.resizeWindow('image', 600,600)
    #         # image = cv2.circle(frame, (int(x*frame.shape[0]),int(y*frame.shape[1])), radius=5, color=(0, 0, 255), thickness=5)
    #
    #         coordinate = df_gaze.iloc[:,count]
    #         for index, pt in enumerate(coordinate):
    #             try:
    #                 (x, y) = ast.literal_eval(pt)
    #                 frame = cv2.circle(frame, (int(x*frame.shape[1]),int(y*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
    #             except Exception as e:
    #                 print(e)
    #             # pt = pt.strip('()')     ## 1315 frame, no gaze point ## 1298
    #             # (x, y) = tuple(map(float, pt.split(', ')))
    #         print(coordinate)
    #         out.write(frame)
    #         cv2.imshow('image', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #         # cv2.waitKey(0)
    #         ret, frame = capture.read()
    #         count += 1
    #     else :
    #         break
