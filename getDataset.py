import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import sys, ast
import numpy as np
import cv2
from tqdm import tqdm
sys.path.append('../')
from variables import Variables

class IMUDataset(Dataset):
    def __init__(self, root, rootfolder, device=None):
        self.root = root
        self.rootfolder = rootfolder
        self.path = self.root + self.rootfolder  + '/' if self.rootfolder[-1]!='/' else (self.root + self.rootfolder)
        self.data = pd.read_csv(self.path + 'imu_file.csv').T
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imu_values = self.data.iloc[:, index][1:]
        data_pts = []
        for index, data in enumerate(imu_values):
            try:
                (acc, gyro) = ast.literal_eval(data)
                data_pt = np.array(acc + gyro)
                data_pt[1] += 9.80665
                data_pts.append(np.round(data_pt, 3))
            except Exception as e:
                print(e)

        return (np.array(data_pts))

class ImageDataset(Dataset):
    def __init__(self, root, rootfolder, video_file='scenevideo.mp4', device=None):
        self.root = root
        self.rootfolder = rootfolder
        self.stack_frames = []
        self.path = self.root + self.rootfolder  + '/' if self.rootfolder[-1]!='/' else (self.root + self.rootfolder)
        self.video_file = self.path + video_file

        self.capture = cv2.VideoCapture(self.video_file)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ret, self.first_frame = self.capture.read()         ## FRAME 1
        self.transforms = transforms.Compose([transforms.ToTensor()])

        # transforms.ToPILImage(),transforms.Resize((512, 512)),

    def __len__(self):
        return 1

    def populate_data(self, last_frame):
        last_frame = cv2.resize(last_frame, (512, 512))
        for frame_num in tqdm(range(10), desc="Building frame dataset"):
            if self.ret == True:
                self.ret, new_frame = self.capture.read()
                new_frame = cv2.resize(new_frame, (512, 512))
                # cv2.imwrite('frame%d.png'.format(frame_num), new_frame)
                stack_frame = np.concatenate((last_frame, new_frame), axis=2)
                stack_frame = self.transforms(stack_frame)
                self.stack_frames.append(stack_frame)

                last_frame = new_frame
            else:
                break

    def __getitem__(self, index):

        return self.stack_frames[index]

if __name__ == "__main__":

    var = Variables()
    print(var.root)
