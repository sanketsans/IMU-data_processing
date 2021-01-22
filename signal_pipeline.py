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

        catIMUData = self.imu_data[imu_index-9]
        for i in range(9):
            catIMUData = np.concatenate((catIMUData, self.imu_data[imu_index-8+i]), axis=0)

        # for i in range(24):
        #     catIMUData = np.concatenate((catIMUData, self.imu_data[imu_index+i]), axis=0)
        return torch.from_numpy(catList).to(self.device), torch.from_numpy(self.gaze_data[index]).to(self.device)
        # return torch.from_numpy(np.concatenate((self.imu_data[imu_index-1], self.imu_data[imu_index]), axis=0)).to(self.device), torch.from_numpy(self.gaze_data[index]).to(self.device)


class IMU_PIPELINE(nn.Module):
    def __init__(self):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.25, bidirectional=True).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        # self.fc2 = nn.Linear(1024, 2).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()

        self.dataset = IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.var.trim_frame_size)
        self.train_imu_dataset, self.test_imu_dataset = self.dataset.imu_train_datasets, self.dataset.imu_test_datasets
        self.train_gaze_dataset, self.test_gaze_dataset = self.dataset.gaze_train_datasets, self.dataset.gaze_test_datasets

        self.loss_fn = nn.MSELoss()
        self.tensorboard_folder = 'BLSTM_signal_outputs_3/'
        self.total_loss, self.current_loss, self.total_accuracy, self.total_correct = 0.0, 10000.0, 0.0, 0

        self.sliced_imu_dataset, self.sliced_gaze_dataset = None, None
        self.unified_dataset = None
        self.gaze_start_index, self.gaze_end_index = 0, 0
        self.imu_start_index, self.imu_end_index = 0, 0
        self.num_samples = 0

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]*1080-label[:,0]*1080) <= 35.0), (torch.abs(pred[:,1]*1920-label[:,1]*1920) <= 60.0)).sum().item()

    def forward(self, x):
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(self.fc1(out[:,-1,:]))
        return out

    def engine(self, sliced_imu_dataset, sliced_gaze_dataset, data_type='train_', optimizer=None):
        self.num_samples = 0
        self.total_loss, self.total_accuracy, self.total_correct = 0.0, 0.0, 0

        self.unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, self.device)
        unified_dataloader = torch.utils.data.DataLoader(self.unified_dataset, batch_size=self.var.batch_size, num_workers=4, drop_last=True)
        tqdm_dataLoader = tqdm(unified_dataloader)
        for batch_index, (imu_data, gaze_data) in enumerate(tqdm_dataLoader):
            self.num_samples += gaze_data.size(0)
            gaze_data = (torch.sum(gaze_data, axis=1) / 4.0)
            coordinates = self.forward(imu_data.float()).to(self.device)
            loss = self.loss_fn(coordinates, gaze_data.float())
            self.total_loss += loss.item()
            self.total_correct += pipeline.get_num_correct(coordinates, gaze_data.float())
            self.total_accuracy = self.total_correct / (coordinates.size(0) * (batch_index+1))
            tqdm_dataLoader.set_description(data_type + '_loss: {:.4} accuracy: {:.3} lowest: {}'.format(
                self.total_loss, self.total_accuracy, self.current_loss))

            if 'train_' in data_type:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.gaze_start_index = self.gaze_end_index
        self.imu_start_index = self.imu_end_index

        return self.total_loss / self.num_samples, self.total_accuracy

if __name__ == "__main__":
    arg = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = 'signal_pipeline_checkpoint.pth'

    n_epochs = 0   ## 250 done, 251 needs to start

    pipeline = IMU_PIPELINE()
    pipeline.dataset = pipeline.prepare_dataset()
    pipeline.train_imu_dataset, pipeline.test_imu_dataset = dataset.imu_train_datasets, dataset.imu_test_datasets
    pipeline.train_gaze_dataset, pipeline.test_gaze_dataset = dataset.gaze_train_datasets, dataset.gaze_test_datasets

    optimizer = optim.Adam(pipeline.parameters(), lr=1e-4)
    print(pipeline)
    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    for epoch in tqdm(range(n_epochs), desc="epochs"):
        pipeline.gaze_start_index, pipeline.imu_start_index = 0, 0
        for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
            loss, accuracy = 0.0, 0.0

            if 'train_' in subDir:
                print(subDir)
                pipeline.train()
                # folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                pipeline.gaze_end_index = pipeline.gaze_start_index + frame_count - pipeline.var.trim_frame_size*2
                pipeline.imu_end_index = pipeline.imu_start_index + frame_count - pipeline.var.trim_frame_size

                sliced_imu_dataset = pipeline.train_imu_dataset[pipeline.imu_start_index: pipeline.imu_end_index]
                sliced_gaze_dataset = pipeline.train_gaze_dataset[pipeline.gaze_start_index: pipeline.gaze_end_index]

                if epoch == 0 and 'del' in arg:
                    _ = os.system('mv runs new_backup')
                    _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                loss, accuracy = pipeline.engine('train_', optimizer)

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
                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    pipeline.gaze_end_index = pipeline.gaze_start_index + frame_count - pipeline.var.trim_frame_size*2
                    pipeline.imu_end_index = pipeline.imu_start_index + frame_count - pipeline.var.trim_frame_size

                    sliced_imu_dataset = pipeline.test_imu_dataset[pipeline.imu_start_index: pipeline.imu_end_index]
                    sliced_gaze_dataset = pipeline.test_gaze_dataset[pipeline.gaze_start_index: pipeline.gaze_end_index]

                    if epoch == 0 and 'del' in arg:
                        _ = os.system('mv runs new_backup')
                        _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                    loss, accuracy = pipeline.engine('test_')

                    tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
                    tb.add_scalar("Loss", loss, epoch)
                    tb.add_scalar("Accuracy", accuracy, epoch)
                    tb.close()

                pipeline.gaze_start_index, pipeline.imu_start_index = 0, 0

        if epoch % 5 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': pipeline.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': pipeline.current_loss
                        }, pipeline.var.root + model_checkpoint)
            print('Model saved')
