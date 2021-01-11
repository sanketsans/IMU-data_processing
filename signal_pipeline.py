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
        return len(self.imu_data) - 1

    def __getitem__(self, index):
        checkedLast = False
        while True:
            check = np.isnan(self.gaze_data[index])
            if check.any():
                index = (index - 1) if checkedLast else (index + 1)
                if index == self.__len__():
                    checkedLast = True
            else:
                break
        return torch.from_numpy(np.concatenate((self.imu_data[index], self.imu_data[index+1]), axis=0)).to(self.device), torch.from_numpy(self.gaze_data[index]*1000.0).to(self.device)


class IMU_PIPELINE(nn.Module):
    def __init__(self, trim_frame_size, device):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = device
        self.trim_frame_size = trim_frame_size
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.2, bidirectional=True).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 1024).to(self.device)
        self.fc2 = nn.Linear(1024, 2).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()

        self.loss_fn = nn.SmoothL1Loss()
        self.tensorboard_folder = 'batch_64_Signal_outputs/'
        self.total_loss, self.current_loss = 0.0, 10000.0
        self.uni_imu_dataset, self.uni_gaze_dataset = None, None
        self.sliced_imu_dataset, self.sliced_gaze_dataset = None, None
        self.unified_dataset = None
        self.start_index, self.end_index = 0, 0

    def prepare_dataset(self):
        self.unified_dataset = IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.trim_frame_size)
        return self.unified_dataset

    def get_num_correct(self, pred, label):
        return (torch.abs(pred - label) <= 10.0).all(axis=1).sum().item()

    def forward(self, x):
        # hidden = (h0, c0)
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.dropout(self.fc1(out[:,-1,:])))
        out = self.activation(self.fc2(out))
        return out*1000.0

    def engine(self, data_type='imu_', optimizer=None):
        self.total_loss = 0.0
        capture = cv2.VideoCapture('scenevideo.mp4')
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.end_index = self.start_index + frame_count - self.trim_frame_size*2

        self.sliced_imu_dataset = self.uni_imu_dataset[self.start_index: self.end_index].detach().cpu().numpy()
        self.sliced_gaze_dataset = self.uni_gaze_dataset[self.start_index: self.end_index].detach().cpu().numpy()
        self.unified_dataset = IMU_DATASET(self.sliced_imu_dataset, self.sliced_gaze_dataset, self.device)

        unified_dataloader = torch.utils.data.DataLoader(self.unified_dataset, batch_size=self.var.batch_size, num_workers=0, drop_last=True)
        tqdm_dataLoader = tqdm(unified_dataloader)
        for batch_index, (imu_data, gaze_data) in enumerate(tqdm_dataLoader):
            gaze_data = (torch.sum(gaze_data, axis=1) / 4.0)
            coordinates = self.forward(imu_data.float()).to(self.device)
            loss = self.loss_fn(coordinates, gaze_data.float())
            self.total_loss += loss.item()
            # total_train_correct += pipeline.get_num_correct(coordinates, gaze_data.float())
            # total_train_accuracy = total_train_correct / (coordinates.size(0) * (batch_index+1))
            tqdm_dataLoader.set_description(data_type + '_loss: {:.4} lr:{:.6} lowest: {}'.format(
                self.total_loss, optimizer.param_groups[0]['lr'],
                self.current_loss))

            if 'imu_' in data_type:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.start_index = self.end_index

        return self.total_loss



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

    optimizer = optim.Adam(pipeline.parameters(), lr=1e-5)
    print(pipeline)

    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    for epoch in tqdm(range(n_epochs), desc="epochs"):
        for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
            train_loss, val_loss, test_loss = 0.0, 0.0, 0.0

            if 'imu_' in subDir:
                pipeline.train()
                # folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                if epoch == 0 and 'del' in arg:
                    _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                train_loss = pipeline.engine('imu_', optimizer)

                if (train_loss < pipeline.current_loss):
                    pipeline.current_loss = train_loss
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': pipeline.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': pipeline.current_loss
                                }, pipeline.var.root + model_checkpoint)
                    print('Model saved')

                pipeline.eval()
                tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
                tb.add_scalar("Loss", train_loss, epoch)
                tb.close()

            if 'val_' in subDir:
                pipeline.eval()
                with torch.no_grad():
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(pipeline.var.root + subDir)
                    if epoch == 0 and 'del' in arg:
                        _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                    val_loss = pipeline.engine('val_')

                    tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
                    tb.add_scalar("Loss", val_loss, epoch)
                    tb.close()

            if 'test_' in subDir:
                pipeline.eval()
                with torch.no_grad():
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(pipeline.var.root + subDir)
                    if epoch == 0 and 'del' in arg:
                        _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                    test_loss = pipeline.engine('test_')
                    tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
                    tb.add_scalar("Loss", test_loss, epoch)
                    tb.close()

        if epoch % 5 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': pipeline.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss
                        }, pipeline.var.root + model_checkpoint)
            print('Model saved')
