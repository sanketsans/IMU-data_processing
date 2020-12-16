import os, json, ast
import matplotlib.pyplot as plt
import sys, math
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import seaborn as sns
sys.path.append('../')
from Pavis_Social_Interaction_Attention_dataset import helpers, variables, proc

panda_data = {}

sample_rate = proc.utils.get_sample_rate(proc.var.timestamps_gaze)
print(sample_rate)
start_index = 0

video_file = 'scenevideo.mp4'
capture = cv2.VideoCapture(video_file)
length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
print(length, fps)
ret, frame = capture.read()
print(frame.shape)

if Path('file.csv').is_file():
    print('File exists')
    df = pd.read_csv('file.csv')

else:
    for sec in range(length):
        panda_data[sec] = list(zip(proc.var.gaze_data[0][start_index:start_index + 4], proc.var.gaze_data[1][start_index:start_index+4]))
        start_index += 4

    df = pd.DataFrame({ key:pd.Series(value) for key, value in panda_data.items()}).T
    df.columns =['Gaze_Pt_1', 'Gaze_Pt_2', 'Gaze_Pt_3', 'Gaze_Pt_4']
    df.to_csv('file.csv')
    df = pd.read_csv('file.csv')

print(len(df))
df = df.T

# frame = cv2.resize(frame, (240, 360), interpolation = cv2.INTER_AREA)
count = 0
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
for i in range(length):
    if ret == True:
        # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 600,600)
        # image = cv2.circle(frame, (int(x*frame.shape[0]),int(y*frame.shape[1])), radius=5, color=(0, 0, 255), thickness=5)

        coordinate = df.iloc[:,count]
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

# coordinate = df.iloc[:, 0]
# for index, pt in enumerate(coordinate):
#     if(index > 0):
#         (x, y) = ast.literal_eval(pt)
#         # pt = pt.strip('()')
#         # (x, y) = tuple(map(float, pt.split(', ')))
#         print(x, y)
#         # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('image', 600,600)
#         image = cv2.circle(frame, (int(x*frame.shape[1]),int(y*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
#         cv2.imshow('image', image)
#         cv2.waitKey(0)

        # plt.imshow(frame)
        # plt.plot(x*240, y*360, "og", markersize=10)
        # plt.plot(x*frame.shape[0], y*frame.shape[1], "og", markersize=10)
