import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split
import argparse
from tqdm import tqdm
sys.path.append('../')
from variables import RootVariables
from helpers import Helpers
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from scipy.signal import butter, lfilter, freqz

class FINAL_DATASET(Dataset):
    def __init__(self, feat, labels):
        self.var = RootVariables()
        self.gaze_data, self.imu_data = [], []
        checkedLast = False
        for index in range(len(labels)):
            check = np.isnan(labels[index])
            imu_check = np.isnan(feat[index])
            if check.any() or imu_check.any():
                continue
            else:
                self.gaze_data.append(labels[index])
                self.imu_data.append(feat[index])

        self.imu_data = self.standarization(self.imu_data)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def standarization(self, datas):
        datas = np.array(datas)
        seq = datas.shape[1]
        datas = datas.reshape(-1, datas.shape[-1])
        rows, cols = datas.shape
        for i in range(cols):
            mean = np.mean(datas[:,i])
            std = np.std(datas[:,i])
            datas[:,i] = (datas[:,i] - mean) / std

        datas = datas.reshape(-1, seq, datas.shape[-1])
        return datas

    def __len__(self):
        return len(self.gaze_data) # len(self.labels)

    def __getitem__(self, index):
        targets = self.gaze_data[index]
        targets[:,0] *= 0.2667
        targets[:,1] *= 0.3556

        return torch.from_numpy(self.imu_data[index]).to(self.device), torch.from_numpy(targets).to(self.device)


class IMU_PIPELINE(nn.Module):
    def __init__(self):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.65, bidirectional=True).to(self.device)
        # self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.dropout = nn.Dropout(0.45)
        self.activation = nn.Sigmoid()

        self.tensorboard_folder = 'signal_Adam1' #'BLSTM_signal_outputs_sell1/'

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0] - label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()
        # return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def forward(self, x):
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        # x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(self.fc1(out[:,-1,:]))
        return out

    def get_original_coordinates(self, pred, labels):
        pred[:,0] *= 3.75*1920.0
        pred[:,1] *= 2.8125*1080.0

        labels[:,0] *= 3.75*1920.0
        labels[:,1] *= 2.8125*1080.0

        return pred, labels

if __name__ == "__main__":
    arg = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_folder = 'test_BookShelf_S1'
    model_checkpoint = 'signal_checkpoint_' + test_folder[5:] + '.pth'

    n_epochs = 0
    toggle = 0

    pipeline = IMU_PIPELINE()
    optimizer = optim.Adam(pipeline.parameters(), lr=1e-4) #, momentum=0.9)
    criterion = nn.L1Loss()
    print(pipeline)
    best_test_loss = 1000.0
    if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_test_loss = checkpoint['best_test_loss']
        # pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    utils = Helpers(test_folder, reset_dataset=0)
    imu_training, imu_testing, training_target, testing_target = utils.load_datasets()
    os.chdir(pipeline.var.root)


    for epoch in tqdm(range(n_epochs), desc="epochs"):
        if epoch > 0:
            utils = Helpers(test_folder, reset_dataset=0)
            imu_training, imu_testing, training_target, testing_target = utils.load_datasets()

        trainDataset = FINAL_DATASET(imu_training_feat, training_target)
        trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
        tqdm_trainLoader = tqdm(trainLoader)
        testDataset = FINAL_DATASET(imu_testing_feat, testing_target)
        testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
        tqdm_testLoader = tqdm(testLoader)

        num_samples = 0
        total_loss, total_correct, total_accuracy = [], 0.0, 0.0
        if epoch == 0 and 'del' in arg:
            # _ = os.system('mv runs new_backup')
            _ = os.system('rm -rf ' + pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)

        trainPD, testPD = [], []
        pipeline.train()
        tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
        for batch_index, (feat, labels) in enumerate(tqdm_trainLoader):
            num_samples += feat.size(0)
            labels = labels[:,0,:]
            pred = pipeline(feat.float()).to(device)
            loss = criterion(pred, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred, labels = pipeline.get_original_coordinates(pred, labels)

                dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
                if batch_index > 0:
                    trainPD = torch.cat((trainPD, dist), 0)
                else:
                    trainPD = dist

                total_loss.append(loss.detach().item())
                total_correct += pipeline.get_num_correct(pred, labels.float())
                total_accuracy = total_correct / num_samples
                tqdm_trainLoader.set_description('training: ' + '_loss: {:.4} correct: {} accuracy: {:.3} MPD: {}'.format(
                    np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(trainPD)))

                if batch_index % 10 :
                    tb.add_scalar("Train Pixel Distance", torch.mean(trainPD[len(trainPD)-10:]), batch_index + (epoch*len(trainLoader)))

        pipeline.eval()
        with torch.no_grad():
            tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
            tb.add_scalar("Train Loss", np.mean(total_loss), epoch)
            tb.add_scalar("Training Correct", total_correct, epoch)
            tb.add_scalar("Train Accuracy", total_accuracy, epoch)
            tb.add_scalar("Mean train pixel dist", torch.mean(trainPD), epoch)

            num_samples = 0
            total_loss, total_correct, total_accuracy = [], 0.0, 0.0
            dummy_correct, dummy_accuracy = 0.0, 0.0
            for batch_index, (feat, labels) in enumerate(tqdm_testLoader):
                num_samples += feat.size(0)
                labels = labels[:,0,:]
                dummy_pts = (torch.ones(8, 2) * 0.5).to(device)
                dummy_pts[:,0] *= 1920
                dummy_pts[:,1] *= 1080

                pred = pipeline(feat.float()).to(device)
                loss = criterion(pred, labels.float())

                pred, labels = pipeline.get_original_coordinates(pred, labels)
                dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
                if batch_index > 0:
                    testPD = torch.cat((testPD, dist), 0)
                else:
                    testPD = dist

                total_loss.append(loss.detach().item())
                total_correct += pipeline.get_num_correct(pred, labels.float())
                dummy_correct += pipeline.get_num_correct(dummy_pts.float(), labels.float())
                dummy_accuracy = dummy_correct / num_samples
                total_accuracy = total_correct / num_samples
                tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3} MPD: {} DAcc: {:.4}'.format(
                    np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(testPD), np.floor(100.0*dummy_accuracy)))

                if batch_index % 10 :
                    tb.add_scalar("Test Pixel Distance", torch.mean(testPD[len(testPD)-10:]), batch_index+(epoch*len(testLoader)))


        tb.add_scalar("Testing Loss", np.mean(total_loss), epoch)
        tb.add_scalar("Testing Correct", total_correct, epoch)
        tb.add_scalar("Testing Accuracy", total_accuracy, epoch)
        tb.add_scalar("Dummy Accuracy", np.floor(100.0*dummy_accuracy), epoch)
        tb.add_scalar("Mean test pixel dist", torch.mean(testPD), epoch)
        tb.close()

        if np.mean(total_loss) <= best_test_loss:
            best_test_loss = np.mean(total_loss)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': pipeline.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_test_loss': best_test_loss,
                        }, pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
            print('Model saved')
