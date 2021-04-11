import numpy as np
import torch
import torch.nn as nn
import os, cv2
import pandas as pd
from ast import literal_eval

if __name__ == '__main__':
    file = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/scenevideo.mp4'
    cap = cv2.VideoCapture(file)
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trim_size = 150
    cap.set(cv2.CAP_PROP_POS_FRAMES,150)
    df = pd.read_csv('/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/gaze_file.csv').to_numpy()
    for i in range(frame_count-300):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = [0.5, 0.5]
        print(df[trim_size+i, 1])
        try:
            gpts = list(map(literal_eval, df[trim_size+i, 1:]))
            avg = [sum(y) / len(y) for y in zip(*gpts)]
            print('avg', avg)
            start_point = (int(pts[0]*frame.shape[1]) - 100, int(pts[1]*frame.shape[0]) + 100)
            end_point = (int(pts[0]*frame.shape[1]) + 100, int(pts[1]*frame.shape[0]) - 100)
            pred_start_point = (int(avg[0]*frame.shape[1]) - 100, int(avg[1]*frame.shape[0]) + 100)
            pred_end_point = (int(avg[0]*frame.shape[1]) + 100, int(avg[1]*frame.shape[0]) - 100)

            frame = cv2.circle(frame, (int(pts[0]*1920),int(pts[1]*1080)), radius=5, color=(0, 0, 255), thickness=5)
            frame = cv2.circle(frame, (int(avg[0]*1920),int(avg[1]*1080)), radius=5, color=(0, 255, 0), thickness=5)

            frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
            frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)
        except:
            pass
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
