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
        self.fc1 = nn.Linear(self.var.hidden_size*2, 512).to(self.device)
        self.fc2 = nn.Linear(512, 2).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()

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

if __name__ == "__main__":
    arg = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = 'signal_pipeline_checkpoint.pth'

    n_epochs = 51
    folders_num = 0
    start_index = 0
    current_loss = 1000.0
    trim_frame_size = 150

    pipeline = IMU_PIPELINE(trim_frame_size, device)

    uni_dataset = pipeline.prepare_dataset()
    uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    uni_gaze_dataset = uni_dataset.gaze_datasets

    optimizer = optim.Adam(pipeline.parameters(), lr=1e-5)
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
            total_train_accuracy, total_val_accuracy, total_test_accuracy = 0.0, 0.0, 0.0
            total_train_correct, total_val_correct, total_test_correct = 0.0, 0.0, 0.0
            capture, frame_count = None, None

            if 'imu_' in subDir:
                pipeline.train()
                # folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                if epoch == 0 and 'del' in arg:
                    _ = os.system('rm -rf runs/Signal_outputs')

                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                end_index = start_index + frame_count - trim_frame_size*2

                sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
                sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)

                unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                tb = SummaryWriter('runs/Signal_outputs/')

                tqdm_trainLoader = tqdm(unified_dataloader)
                for batch_index, (imu_data, gaze_data) in enumerate(tqdm_trainLoader):
                    gaze_data = (torch.sum(gaze_data, axis=1) / 4.0)
                    optimizer.zero_grad()
                    coordinates = pipeline(imu_data.float()).to(device)
                    loss = loss_fn(coordinates, gaze_data.float())
                    train_loss += loss.item()
                    total_train_correct += pipeline.get_num_correct(coordinates, gaze_data.float())
                    total_train_accuracy = total_train_correct / (coordinates.size(0) * (batch_index+1))
                    tqdm_trainLoader.set_description('loss: {:.4} lr:{:.6} lowest: {}'.format(
                        train_loss, optimizer.param_groups[0]['lr'],
                        current_loss))

                    loss.backward()
                    optimizer.step()

                if (train_loss < current_loss):
                    current_loss = train_loss
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': pipeline.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': current_loss
                                }, pipeline.var.root + model_checkpoint)
                    print('Model saved')

                start_index = end_index
                pipeline.eval()
                tb.add_scalar("Loss", train_loss, epoch)
                tb.close()
                with open(pipeline.var.root + 'signal_train_loss.txt', 'a') as f:
                    f.write(str(train_loss) + '\n')
                    f.close()

            if 'val_' in subDir:
                pipeline.eval()
                with torch.no_grad():
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(pipeline.var.root + subDir)
                    if epoch == 0 and 'del' in arg:
                        _ = os.system('rm -rf runs/Signal_outputs')

                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    end_index = start_index + frame_count - trim_frame_size*2
                    sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
                    sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                    unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
                    unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                    tb = SummaryWriter('runs/Signal_outputs/')

                    tqdm_valLoader = tqdm(unified_dataloader)
                    for batch_index, (imu_data, gaze_data) in enumerate(tqdm_valLoader):
                        gaze_data = (torch.sum(gaze_data, axis=1) / 4.0)
                        coordinates = pipeline(imu_data.float()).to(device)
                        loss = loss_fn(coordinates, gaze_data.float())
                        val_loss += loss.item()
                        total_val_correct += pipeline.get_num_correct(coordinates, gaze_data.float())
                        total_val_accuracy = total_val_correct / (coordinates.size(0) * (batch_index+1))
                        tqdm_valLoader.set_description('loss: {:.4} lr:{:.6} '.format(
                            val_loss,optimizer.param_groups[0]['lr']))

                    tb.add_scalar("Loss", val_loss, epoch)
                    tb.close()

                    with open(pipeline.var.root + 'signal_validation_loss.txt', 'a') as f:
                        f.write(str(val_loss) + '\n')
                        f.close()

                start_index = end_index

            if 'test_' in subDir:
                pipeline.eval()
                with torch.no_grad():
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(pipeline.var.root + subDir)
                    if epoch == 0 and 'del' in arg:
                        _ = os.system('rm -rf runs/Signal_outputs')
                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    end_index = start_index + frame_count - trim_frame_size*2
                    sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
                    sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                    unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
                    unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                    tb = SummaryWriter('runs/Signal_outputs/')

                    tqdm_testLoader = tqdm(unified_dataloader)
                    for batch_index, (imu_data, gaze_data) in enumerate(tqdm_testLoader):
                        gaze_data = (torch.sum(gaze_data, axis=1) / 4.0)
                        coordinates = pipeline(imu_data.float()).to(device)
                        loss = loss_fn(coordinates, gaze_data.float())
                        test_loss += loss.item()
                        total_test_correct += pipeline.get_num_correct(coordinates, gaze_data.float())
                        total_test_accuracy = total_test_correct / (coordinates.size(0) * (batch_index + 1))
                        tqdm_testLoader.set_description('loss: {:.4} lr:{:.6}'.format(
                            test_loss, optimizer.param_groups[0]['lr']))

                    tb.add_scalar("Loss", test_loss, epoch)
                    tb.close()

                    with open(pipeline.var.root + 'signal_testing_loss.txt', 'a') as f:
                        f.write(str(test_loss) + '\n')
                        f.close()

                start_index = end_index

        if epoch % 5 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': pipeline.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss
                        }, pipeline.var.root + model_checkpoint)
            print('Model saved')
