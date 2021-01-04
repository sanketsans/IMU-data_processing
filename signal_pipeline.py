import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import argparse
from tqdm import tqdm
sys.path.append('../')
from prepare_dataset import IMU_GAZE_FRAME_DATASET
from variables import RootVariables

class IMU_DATASET(Dataset):
    def __init__(self, imu_data, gaze_data, device=None):
        self.imu_data = imu_data
        self.gaze_data = gaze_data
        self.device = device

    def __len__(self):
        return len(self.imu_data) - 1

    def __getitem__(self, index):
        return torch.from_numpy(self.imu_data[index]).to(self.device), torch.from_numpy(self.gaze_data[index]).to(self.device)


class IMU_PIPELINE(nn.Module):
    def __init__(self, trim_frame_size, device):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = device
        self.trim_frame_size = trim_frame_size
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.2, bidirectional=True).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 128).to(self.device)
        self.fc2 = nn.Linear(128, 2).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()

    def prepare_dataset(self):
        self.unified_dataset = IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.trim_frame_size)
        return self.unified_dataset

    def forward(self, x):
        # hidden = (h0, c0)
        h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        print(out[:, :2, :].shape)
        out = F.relu(self.dropout(self.fc1(out[:, -1, :])))
        out = self.activation(self.fc2(out))
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = 'signal_pipeline_checkpoint.pth'

    n_epochs = 0
    folders_num = 0
    start_index = 0
    current_loss = 1000.0
    trim_frame_size = 150

    pipeline = IMU_PIPELINE(trim_frame_size, device)

    uni_dataset = pipeline.prepare_dataset()
    uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    uni_gaze_dataset = uni_dataset.gaze_datasets

    set_a = uni_imu_dataset[0: 1493, 2]
    set_b = uni_imu_dataset[0: 1493, 3]
    print(torch.max(set_a), torch.min(set_a))
    print(torch.max(set_b), torch.min(set_b))

    # plt.plot(x, set_a, 'r')
    # plt.plot(x, set_b, 'g')


    optimizer = optim.Adam(pipeline.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss()
    print(pipeline)

    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_loss = checkpoint['loss']
        print('Model loaded')

    for epoch in tqdm(range(n_epochs), desc="epochs"):
        start_index = 0
        for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
            unified_dataset, unified_dataloader = None, None
            sliced_imu_dataset, sliced_gaze_dataset = None, None
            train_loss, val_loss, test_loss = 0.0, 0.0, 0.0
            capture, frame_count = None, None

            if 'imu_' in subDir:
                pipeline.train()
                # folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                end_index = start_index + frame_count - trim_frame_size*2

                sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
                sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)

                unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)

                tqdm_trainLoader = tqdm(unified_dataloader)
                for batch_index, (imu_data, gaze_data) in enumerate(tqdm_trainLoader):
                    gaze_data = torch.sum(gaze_data, axis=1) / 4.0
                    coordinates = pipeline(imu_data.float()).to(device)
                    print('gaze: ', gaze_data.shape)
            #         loss = loss_fn(coordinates, gaze_data.float())
            #         train_loss += loss.item()
            #         tqdm_trainLoader.set_description('loss: {:.4} lr:{:.6} lowest: {}'.format(
            #             loss.item(), optimizer.param_groups[0]['lr'], current_loss))
            #
            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
            #
            #     if ((train_loss/len(unified_dataloader)) < current_loss):
            #         current_loss = (train_loss/len(unified_dataloader))
            #         torch.save({
            #                     'epoch': epoch,
            #                     'model_state_dict': pipeline.state_dict(),
            #                     'optimizer_state_dict': optimizer.state_dict(),
            #                     'loss': current_loss
            #                     }, pipeline.var.root + model_checkpoint)
            #         print('Model saved')
            #
            #     start_index = end_index
            #     with open(pipeline.var.root + 'signal_train_loss.txt', 'a') as f:
            #         f.write(str(train_loss/len(unified_dataloader)) + '\n')
            #         f.close()
            #
            # if 'val_' in subDir:
            #     pipeline.eval()
            #     with torch.no_grad():
            #         subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            #         os.chdir(pipeline.var.root + subDir)
            #         capture = cv2.VideoCapture('scenevideo.mp4')
            #         frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            #         end_index = start_index + frame_count - trim_frame_size*2
            #         sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
            #         sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()
            #
            #         unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
            #         unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
            #         tqdm_valLoader = tqdm(unified_dataloader)
            #         for batch_index, (imu_data, gaze_data) in enumerate(tqdm_valLoader):
            #             gaze_data = torch.sum(gaze_data, axis=1) / float(gaze_data.shape[1])
            #             coordinates = pipeline(imu_data.float()).to(device)
            #             loss = loss_fn(coordinates, gaze_data.float())
            #             val_loss += loss.item()
            #             tqdm_valLoader.set_description('loss: {:.4} lr:{:.6}'.format(
            #                 loss.item(), optimizer.param_groups[0]['lr']))
            #
            #         with open(pipeline.var.root + 'signal_validation_loss.txt', 'a') as f:
            #             f.write(str(val_loss/len(unified_dataloader)) + '\n')
            #             f.close()
            #
            #     start_index = end_index
            #
            # if 'test_' in subDir:
            #     pipeline.eval()
            #     with torch.no_grad():
            #         subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            #         os.chdir(pipeline.var.root + subDir)
            #         capture = cv2.VideoCapture('scenevideo.mp4')
            #         frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            #         end_index = start_index + frame_count - trim_frame_size*2
            #         sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
            #         sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()
            #
            #         unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
            #         unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
            #         tqdm_testLoader = tqdm(unified_dataloader)
            #         for batch_index, (imu_data, gaze_data) in enumerate(tqdm_testLoader):
            #             gaze_data = torch.sum(gaze_data, axis=1) / 4.0
            #             coordinates = pipeline(imu_data.float()).to(device)
            #             loss = loss_fn(coordinates, gaze_data.float())
            #             test_loss += loss.item()
            #             tqdm_testLoader.set_description('loss: {:.4} lr:{:.6}'.format(
            #                 loss.item(), optimizer.param_groups[0]['lr']))
            #
            #         with open(pipeline.var.root + 'signal_testing_loss.txt', 'a') as f:
            #             f.write(str(test_loss/len(unified_dataloader)) + '\n')
            #             f.close()
            #
            #     start_index = end_index

        if epoch % 5 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': pipeline.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss
                        }, pipeline.var.root + model_checkpoint)
            print('Model saved')
