import os, cv2
from tqdm import tqdm
import torch, argparse
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pipeline_new import FusionPipeline, FINAL_DATASET
from helpers import Helpers
from variables import RootVariables

if __name__ == "__main__":

    var = RootVariables()
    folder = 'train_PosterSession_S3'
    utils = Helpers('train_Lift_S1')
    _, imu, targets = utils.load_datasets_folder(folder)
    print('tuils: ', utils.test_gaze_dataset[0])
    print(targets[1])

    os.chdir(var.root + folder)
    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES,var.trim_frame_size)
    ret, frame = capture.read()

    for i in range(0):
        if ret == True:
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 512, 512)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # coordinate = sliced_gaze_dataset[i]
            try:
                gt_gaze_pts = targets[i][0]
                # gt_gaze_pts = np.sum(sliced_gaze_dataset[i], axis=0) / 4.0
                # pred_gaze_pts = coordinate[i]
                padding_r = 100.0
                padding = 100.0
                # plt.scatter(int(pred_gaze_pts[0]*frame.shape[1]), int(pred_gaze_pts[1]*frame.shape[0]))

                start_point = (int(gt_gaze_pts[0]*frame.shape[1]) - int(padding), int(gt_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                end_point = (int(gt_gaze_pts[0]*frame.shape[1]) + int(padding), int(gt_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                # pred_start_point = (int(pred_gaze_pts[0]*frame.shape[1]) - int(padding), int(pred_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                # pred_end_point = (int(pred_gaze_pts[0]*frame.shape[1]) + int(padding), int(pred_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                #
                frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
                # frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)

                frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1]),int(gt_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
                # frame = cv2.circle(frame, (int(pred_gaze_pts[0]*frame.shape[1]),int(pred_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 255, 0), thickness=5)
            except Exception as e:
                print(e)
            cv2.imshow('image', frame)
            # out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            ret, frame = capture.read()
