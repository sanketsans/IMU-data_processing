import os, json
import matplotlib.pyplot as plt
import sys, math
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import seaborn as sb
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

if Path('file.csv').is_file():
    print('File exists')
    df = pd.read_csv('file.csv')

else:
    for sec in range(length):
        panda_data[sec] = list(zip(proc.var.gaze_data[0][start_index:start_index + 4], proc.var.gaze_data[1][start_index:start_index+4]))
        start_index += 4

    df = pd.DataFrame({ key:pd.Series(value) for key, value in panda_data.items() }).T
    df.to_csv('file.csv')

print(df.head)
# data = np.random.rand(4,6)
# heatmap = sb.heatmap(data)
proc.plt.imshow(df, cmap ="RdYlBu")
proc.plt.show()
# while ret:
#   cv2.imwrite("frames/frame%d.jpg" % count, frame)     # save frame as JPEG file
#   ret,frame = capture.read()
#   print('Read a new frame: ', ret)
#   count += 1
