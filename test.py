import os, cv2, sys
from tqdm import tqdm
import torch, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pipeline_new import FusionPipeline
from signal_pipeline import IMU_PIPELINE, IMU_DATASET
from prepare_dataset import IMU_GAZE_FRAME_DATASET, UNIFIED_DATASET
from torch.utils.tensorboard import SummaryWriter

class TEST_DATASET(Dataset):
    def __init__(self, imudata, gazedata):
        self.imudata = imudata
        self.gazedata = gazedata

    def __len__(self):
        return len(self.gazedata) -1

    def __getitem__(self, index):
        checkedLast = False
        while True:
            check = np.isnan(self.gazedata[index])
            if check.any():
                index = (index - 1) if checkedLast else (index + 1)
                if index == self.__len__():
                    checkedLast = True
            else:
                break
        return self.imudata[index], self.gazedata[index]

if __name__ == "__main__":
    # askindex = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trim_frame_size = 150
    model_checkpoint = 'signal_pipeline_checkpoint.pth'
    pipeline = IMU_PIPELINE(trim_frame_size, device)

    uni_dataset = pipeline.prepare_dataset()
    uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    uni_gaze_dataset = uni_dataset.gaze_datasets


    start_index, end_index = 0, 0

    pipeline.eval()
    sliced_imu_dataset, sliced_gaze_dataset, sliced_frame_dataset = None, None, None
    catList = None
    for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
        if 'imu_' in subDir:
            print(subDir)
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_index = start_index + frame_count - trim_frame_size*2
            sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
            sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()
            dataset = TEST_DATASET(sliced_imu_dataset, sliced_gaze_dataset)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)

            start_index = end_index

        if 'test_' in subDir:
            print(subDir)
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_index = start_index + frame_count - trim_frame_size*2
            sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
            sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()
            start_index = end_index

        if 'val_' in subDir:
            print(subDir)
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_index = start_index + frame_count - trim_frame_size*2
            sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
            sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()
            start_index = end_index

        if 'imu_' in subDir or 'val_' in subDir or 'test_' in subDir:
            print(sliced_imu_dataset[0], sliced_gaze_dataset[0])
            print(sliced_imu_dataset[-1], sliced_gaze_dataset[-1])
            # fig = plt.figure()
            # sliced_imu_dataset = sliced_imu_dataset.reshape(-1, 6)
            # fig.add_subplot(221)
            # plt.hist(sliced_imu_dataset[:,0], bins='auto')
            # fig.add_subplot(222)
            # plt.hist(sliced_imu_dataset[:,1], bins='auto')
            # fig.add_subplot(223)
            # plt.hist(sliced_imu_dataset[:,2], bins='auto')
            # fig.add_subplot(221)
            # plt.hist(sliced_imu_dataset[:,3], bins='auto')
            # fig.add_subplot(222)
            # plt.hist(sliced_imu_dataset[:,4], bins='auto')
            # fig.add_subplot(223)
            # plt.hist(sliced_imu_dataset[:,5], bins='auto')
            # plt.show()



        # dataset = TEST_DATASET(sliced_imu_dataset, sliced_gaze_dataset)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)
        # for index, (imu, gaze) in enumerate(dataloader):
        #     print('BATCH ITER: ', index, gaze)

    # sliced_gaze_dataset[40][where] = np.mean(sliced_gaze_dataset[40][not where].all())
    # print(sliced_gaze_dataset[40][:,0], where)
    # df = pd.DataFrame(sliced_gaze_dataset.reshape(-1, 2))
    # df = df[df['EPS'].notna()]

    # df = df[df[0].notna()]
    # df.to_csv('test.csv')
    # print(df[0])
