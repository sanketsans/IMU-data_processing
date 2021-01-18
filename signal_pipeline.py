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
from torch.utils.tensorboard import SummaryWriter

class IMU_DATASET(Dataset):
    def __init__(self, imu_data, gaze_data, device=None):
        self.imu_data = imu_data
        self.gaze_data = gaze_data
        self.device = device

    def __len__(self):
        return len(self.gaze_data) - 1

    def __getitem__(self, index):
        print(index, len(self.imu_data))
        imu_index = 75 + index
        checkedLast = False
        catIMUData = None
        while True:
            check = np.isnan(self.gaze_data[index])
            if check.any():
                index = (index - 1) if checkedLast else (index + 1)
                if index == self.__len__():
                    checkedLast = True
            else:
                break

        catIMUData = self.imu_data[imu_index-25]
        for i in range(25):
            catIMUData = np.concatenate((catIMUData, self.imu_data[imu_index-24+i]), axis=0)

        for i in range(24):
            catIMUData = np.concatenate((catIMUData, self.imu_data[imu_index+i]), axis=0)

        return torch.from_numpy(catIMUData).to(self.device), torch.from_numpy(self.gaze_data[index]*1000.0).to(self.device)


class IMU_PIPELINE(nn.Module):
    def __init__(self, trim_frame_size, device):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = device
        self.trim_frame_size = trim_frame_size
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.2, bidirectional=True).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.fc0 = nn.Linear(2, self.var.imu_input_size).to(self.device)
        # self.fc2 = nn.Linear(1024, 2).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()

        self.loss_fn = nn.SmoothL1Loss()
        self.tensorboard_folder = 'BLSTM_signal_outputs_2//'
        self.total_loss, self.current_loss, self.total_accuracy, self.total_correct = 0.0, 10000.0, 0.0, 0
        self.uni_imu_dataset, self.uni_gaze_dataset = None, None
        self.sliced_imu_dataset, self.sliced_gaze_dataset = None, None
        self.unified_dataset = None
        self.gaze_start_index, self.gaze_end_index = 0, 0
        self.imu_start_index, self.imu_end_index = 0, 0

    def prepare_dataset(self):
        self.unified_dataset = IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.trim_frame_size)
        return self.unified_dataset

    def get_num_correct(self, pred, label):
        return (torch.abs(pred - label) <= 30.0).all(axis=1).sum().item()

    def init_stage(self):
        # IMU Model
        self.imuModel_h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        self.imuModel_c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

    def forward(self, x):
        # hidden = (h0, c0)
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        x = F.relu(self.fc0(x))
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.activation(self.fc1(out[:,-1,:])))
        return out*1000.0

    def engine(self, data_type='imu_', optimizer=None):
        self.total_loss, self.total_accuracy, self.total_correct = 0.0, 0.0, 0
        capture = cv2.VideoCapture('scenevideo.mp4')
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.gaze_end_index = self.gaze_start_index + frame_count - self.trim_frame_size*2
        self.imu_end_index = self.imu_start_index + frame_count - self.trim_frame_size

        self.sliced_imu_dataset = self.uni_imu_dataset[self.imu_start_index: self.imu_end_index]
        self.sliced_gaze_dataset = self.uni_gaze_dataset[self.gaze_start_index: self.gaze_end_index]
        self.unified_dataset = IMU_DATASET(self.sliced_imu_dataset, self.sliced_gaze_dataset, self.device)
        unified_dataloader = torch.utils.data.DataLoader(self.unified_dataset, batch_size=self.var.batch_size, num_workers=0, drop_last=True)
        tqdm_dataLoader = tqdm(unified_dataloader)
        for batch_index, (imu_data, gaze_data) in enumerate(tqdm_dataLoader):

            gaze_data = (torch.sum(gaze_data, axis=1) / 4.0)
            coordinates = self.forward(imu_data.float()).to(self.device)
            loss = self.loss_fn(coordinates, gaze_data.float())
            self.total_loss += loss.item()
            self.total_correct += pipeline.get_num_correct(coordinates, gaze_data.float())
            self.total_accuracy = self.total_correct / (coordinates.size(0) * (batch_index+1))
            tqdm_dataLoader.set_description(data_type + '_loss: {:.4} accuracy: {:.3} lowest: {}'.format(
                self.total_loss, self.total_accuracy, self.current_loss))

            if 'imu_' in data_type:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.gaze_start_index = self.gaze_end_index
        self.imu_start_index = self.imu_end_index

        return self.total_loss, self.total_accuracy

if __name__ == "__main__":
    arg = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = 'signal_pipeline_checkpoint.pth'

    n_epochs = 1   ## 250 done, 251 needs to start
    trim_frame_size = 150

    pipeline = IMU_PIPELINE(trim_frame_size, device)

    uni_dataset = pipeline.prepare_dataset()
    pipeline.uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    pipeline.uni_gaze_dataset = uni_dataset.gaze_datasets
    print(len(pipeline.uni_imu_dataset))

    optimizer = optim.Adam(pipeline.parameters(), lr=1e-4)
    print(pipeline)
    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    for epoch in tqdm(range(n_epochs), desc="epochs"):
        pipeline.start_index = 0
        for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
            loss, accuracy = 0.0, 0.0

            if 'imu_' in subDir:
                print(subDir)
                pipeline.train()
                # folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                if epoch == 0 and 'del' in arg:
                    _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                loss, accuracy = pipeline.engine('imu_', optimizer)

                if (loss < pipeline.current_loss):
                    pipeline.current_loss = loss
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': pipeline.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': pipeline.current_loss
                                }, pipeline.var.root + model_checkpoint)
                    print('Model saved')

                pipeline.eval()
                tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
                tb.add_scalar("Loss", loss, epoch)
                tb.add_scalar("Accuracy", accuracy, epoch)
                tb.close()

            if 'val_' in subDir or 'test_' in subDir:
                print(subDir)
                pipeline.eval()
                with torch.no_grad():
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(pipeline.var.root + subDir)
                    if epoch == 0 and 'del' in arg:
                        _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                    loss, accuracy = pipeline.engine('test_')

                    tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
                    tb.add_scalar("Loss", loss, epoch)
                    tb.add_scalar("Accuracy", accuracy, epoch)
                    tb.close()

        if epoch % 5 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': pipeline.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': pipeline.current_loss
                        }, pipeline.var.root + model_checkpoint)
            print('Model saved')
