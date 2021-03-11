import os, cv2
from tqdm import tqdm
import torch, argparse
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from helpers import Helpers
from variables import RootVariables
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.transform import rotate
import math

### PLOTTNG GAZE OUTPUTS
if __name__ == "__main__":

    var = RootVariables()
    folder = 'train_InTheDeak_S2'
    utils = Helpers('test_BookShelf_S1', reset_dataset=0)
    targets = utils.load_datasets_folder(folder)
    print(targets[0])

    os.chdir(var.root + folder)
    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES,var.trim_frame_size)
    ret, frame = capture.read()
    print(frame.shape)
    # plt.scatter(0, 0)
    # plt.scatter(1920, 0)
    # plt.scatter(0, 1080)
    # plt.scatter(1920, 1080)
    # file = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/testing_images/frames_0.npy'
    # img = np.load(file)
    # print(img.shape)
    # img = rotate(img, angle=10)
    # img = img[:,:,:3]
    # plt.imshow(img)
    # plt.show()
    for i in range(frame_count - 300):
        if ret == True:
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 384, 512)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # coordinate = sliced_gaze_dataset[i]
            try:
                gt_gaze_pts = targets[i][0]
                # gt_gaze_pts[0] *= 0.2667
                # gt_gaze_pts[1] *= 0.3556
                gt_gaze_pts[0] -= 0.5
                gt_gaze_pts[1] -= 0.5

                # gt_gaze_pts = np.sum(sliced_gaze_dataset[i], axis=0) / 4.0
                # pred_gaze_pts = coordinate[i]
                padding_r = 100.0
                padding = 100.0
                # plt.scatter(int(pred_gaze_pts[0]*frame.shape[1]), int(pred_gaze_pts[1]*frame.shape[0]))

                # start_point = (int(gt_gaze_pts[0]*frame.shape[1]) - int(padding), int(gt_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                # end_point = (int(gt_gaze_pts[0]*frame.shape[1]) + int(padding), int(gt_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                # pred_start_point = (int(pred_gaze_pts[0]*frame.shape[1]) - int(padding), int(pred_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                # pred_end_point = (int(pred_gaze_pts[0]*frame.shape[1]) + int(padding), int(pred_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                #
                # frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
                # frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5

                original = np.copy(gt_gaze_pts)
                angle = 20

                gt_gaze_pts[0] = original[0]*(math.cos(math.radians(angle))) + original[1]*(math.sin(math.radians(angle)))
                gt_gaze_pts[1] = original[1]*(math.cos(math.radians(angle))) - original[0]*(math.sin(math.radians(angle)))
                # original = np.copy(gt_gaze_pts)
                # angle = -20
                #
                # gt_gaze_pts[0] = original[0]*(math.cos(math.radians(angle))) + original[1]*(math.sin(math.radians(angle)))
                # gt_gaze_pts[1] = original[1]*(math.cos(math.radians(angle))) - original[0]*(math.sin(math.radians(angle)))
                gt_gaze_pts[0] += 0.5
                gt_gaze_pts[1] += 0.5
                #
                # gt_gaze_pts[0] *= 3.75*1920
                # gt_gaze_pts[1] *= 2.8125*1080

                gt_gaze_pts[0] *= 512.0
                gt_gaze_pts[1] *= 384.0

                gt_gaze_pts[0] *= 3.75
                gt_gaze_pts[1] *= 2.8125

                # frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1]),int(gt_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
                # frame = cv2.resize(frame, (512, 384))
                frame = rotate(frame, angle=angle)
                frame = cv2.circle(frame, (int(gt_gaze_pts[0]),int(gt_gaze_pts[1])), radius=5, color=(0, 0, 255), thickness=5)
                # plt.scatter(int(gt_gaze_pts[0]*frame.shape[1]),int(gt_gaze_pts[1]*frame.shape[0]))
                # frame = cv2.circle(frame, (int(pred_gaze_pts[0]*frame.shape[1]),int(pred_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 255, 0), thickness=5)
            except Exception as e:
                print(e)
            cv2.imshow('image', frame)
            # out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            ret, frame = capture.read()

    # plt.show()

# BookShelf_S1​
# CoffeeVendingMachine_S1​
# CoffeeVendingMachine_S2​
# CoffeeVendingMachine_S3​
# InTheDeak_S1​
# InTheDeak_S2​
# Lift_S1​
# NespressoCoffeeMachine_S1​
# NespressoCoffeeMachine_S2​
# Outdoor_S1​
# PosterSession_S1​
# PosterSession_S2​
# PosterSession_S3​
# PosterSession_S4​
# smallGroupMeeting_S1​
# smallGroupMeeting_S2​
# smallGroupMeeting_S3​
# smallGroupMeeting_S3​

# class FINAL_DATASET(Dataset):
#     def __init__(self, feat, labels):
#         self.feat = feat
#         self.label = labels
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         print(index)
#         return self.feat[index], self.label[index]
#
#
# if __name__ == '__main__':
#     var = RootVariables()
#     dir = var.root + 'test_Lift_S1/images/'
#     _ = os.chdir(dir)
#     # list = os.listdir(dir)
#     # print(len(list))
#
#     img1 = cv2.imread('frames_0.jpg')
#     img2 = Image.open('frames_1.jpg')
#     transforms = transforms.Compose([transforms.RandomCrop(256), transforms.ToTensor()])
#     print(transforms(img1).shape)
