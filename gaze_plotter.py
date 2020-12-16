import os, json, ast
import matplotlib.pyplot as plt
import sys, math
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import seaborn as sns
sys.path.append('../')
from gaze_data import helpers, variables, proc

panda_data = {}

# sample_rate = proc.utils.get_sample_rate(proc.var.timestamps_gaze)
# sample_rate = proc.utils.get_sample_rate(proc.var.timestamps_imu)
# print(sample_rate)
start_index = 0

video_file = 'scenevideo.mp4'
capture = cv2.VideoCapture(video_file)
length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
print(length, fps)
ret, frame = capture.read()
print(frame.shape)

if Path('gaze_file.csv').is_file():
    print('File exists')
    df_gaze = pd.read_csv('gaze_file.csv')
    df_imu = pd.read_csv('imu_file.csv')
    # os.system('rm file.csv')
    # print('Deleted file')

else:
    ## IMU DATAFRAME
    for sec in range(length):
        panda_data[sec] = list(zip(proc.var.gaze_data[0][start_index:start_index + 4], proc.var.gaze_data[1][start_index:start_index+4]))
        start_index += 4

    df_gaze = pd.DataFrame({ key:pd.Series(value) for key, value in panda_data.items()}).T
    df_gaze.columns =['Gaze_Pt_1', 'Gaze_Pt_2', 'Gaze_Pt_3', 'Gaze_Pt_4']
    df_gaze.to_csv('gaze_file.csv')
    df_gaze = pd.read_csv('gaze_file.csv')

    ## GAZE DATAFRAME
    for sec in range(length):
        panda_data[sec] = list(tuple((sec, sec+2)))
        panda_data[sec] = list(zip(zip(proc.var.imu_data_acc[0][start_index:start_index+4],
                                    proc.var.imu_data_acc[1][start_index:start_index+4],
                                    proc.var.imu_data_acc[2][start_index:start_index+4]),

                                zip(proc.var.imu_data_gyro[0][start_index:start_index+4],
                                        proc.var.imu_data_gyro[1][start_index:start_index+4],
                                        proc.var.imu_data_gyro[2][start_index:start_index+4])))
        # panda_data[sec] = list(zip(proc.var.imu_data_acc[0][start_index:start_index + 4], proc.var.imu_data_gyro[0][start_index:start_index+4]))
        start_index += 4

    df_imu = pd.DataFrame({ key:pd.Series(value) for key, value in panda_data.items()}).T
    df_imu.columns =['IMU_Acc/Gyro_Pt_1', 'IMU_Acc/Gyro_Pt_2', 'IMU_Acc/Gyro_Pt_3', 'IMU_Acc/Gyro_Pt_4']
    df_imu.to_csv('imu_file.csv')
    df_imu = pd.read_csv('imu_file.csv')


print(len(df_gaze), len(df_imu))
df_gaze = df_gaze.T

count = 0
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
for i in range(length):
    if ret == True:
        # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 600,600)
        # image = cv2.circle(frame, (int(x*frame.shape[0]),int(y*frame.shape[1])), radius=5, color=(0, 0, 255), thickness=5)

        coordinate = df_gaze.iloc[:,count]
        for index, pt in enumerate(coordinate):
            try:
                (x, y) = ast.literal_eval(pt)
                frame = cv2.circle(frame, (int(x*frame.shape[1]),int(y*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
            except Exception as e:
                print(e)
            # pt = pt.strip('()')     ## 1315 frame, no gaze point ## 1298
            # (x, y) = tuple(map(float, pt.split(', ')))
        print(coordinate)
        out.write(frame)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.waitKey(0)
        ret, frame = capture.read()
        count += 1
    else :
        break
