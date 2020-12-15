import os
import json
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import pandas as pd
import cv2
sys.path.append('../')
from Pavis_Social_Interaction_Attention_dataset import helpers, variables, proc

panda_data = {}

sample_rate = proc.utils.get_sample_rate(proc.var.timestamps_gaze)
print(sample_rate)
start_index = 0

for sec in sample_rate:
    panda_data[sec] = list(zip(proc.var.gaze_data[0][start_index:start_index + sample_rate[sec]], proc.var.gaze_data[1][start_index:start_index+sample_rate[sec]]))
    start_index += sample_rate[sec]

df = pd.DataFrame({ key:pd.Series(value) for key, value in panda_data.items() })

print(df.head)
df.to_csv('file.csv')
# dataframe = pd.DataFrame(panda_data)
# print(dataframe)

video_file = 'scenevideo.mp4'
capture = cv2.VideoCapture(video_file)
length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
print(length, fps)
start = 0.0
count = 0
for i in range(0, length):
    ret, frame = capture.read()
    count += 0
    height, width = frame.shape[:2]
    print(height, width)
    break
