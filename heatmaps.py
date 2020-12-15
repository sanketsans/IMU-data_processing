import os, json
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
    # df.to_csv('file.csv')

df = df.T
# frame = cv2.resize(frame, (240, 360), interpolation = cv2.INTER_AREA)
count = 0
for i in range(length):
    while ret:
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)
        # image = cv2.circle(frame, (int(x*frame.shape[0]),int(y*frame.shape[1])), radius=5, color=(0, 0, 255), thickness=5)
        coordinate = df.iloc[:,count]
        for index, pt in enumerate(coordinate):
            if index > 0:
                (x, y) = pt
                # pt = pt.strip('()')     ## 1315 frame, no gaze point ## 1298
                # (x, y) = tuple(map(float, pt.split(', ')))
                image = cv2.circle(frame, (int(x*frame.shape[1]),int(y*frame.shape[0])), radius=8, color=(0, 0, 255), thickness=10)
        print(coordinate)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        ret, frame = capture.read()
        count += 1

# for index, pt in enumerate(coordinate):
#     if(index > 0):
#         pt = pt.strip('()')
#         (x, y) = tuple(map(float, pt.split(', ')))
#         cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('image', 600,600)
#         image = cv2.circle(frame, (int(x*frame.shape[0]),int(y*frame.shape[1])), radius=5, color=(0, 0, 255), thickness=5)
#         cv2.imshow('image', image)
#         cv2.waitKey(0)

        # plt.imshow(frame)
        # plt.plot(x*240, y*360, "og", markersize=10)
        # plt.plot(x*frame.shape[0], y*frame.shape[1], "og", markersize=10)

# plt.show()
# cv2.destroyAllWindows(0)
# print(len(coordinate), coordinate.Gaze_Pt_1)
# plt.imshow(frame)
# plt.plot(x*frame.shape[0], y*frame.shape[1], "og", markersize=10)
# plt.show()
# print(x*frame.shape[0], y*frame.shape[1])
# data = np.random.rand(4,6)
# heatmap = sb.heatmap(data)
# proc.plt.imshow(df, cmap ="RdYlBu")
# proc.plt.show()
# while ret:
#   cv2.imwrite("frames/frame%d.jpg" % count, frame)     # save frame as JPEG file
#   ret,frame = capture.read()
#   print('Read a new frame: ', ret)
#   count += 1
