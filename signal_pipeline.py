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
        self.feat = feat
        self.label = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return torch.from_numpy(self.feat[index]).to(self.device), torch.from_numpy(self.label[index]).to(self.device)


class IMU_PIPELINE(nn.Module):
    def __init__(self):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.65, bidirectional=True).to(self.device)
        self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()

        self.tensorboard_folder = sys.argv[2] #'BLSTM_signal_outputs_sell1/'

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

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
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        h0 = Variable(h0, requires_grad=True)
        c0 = Variable(c0, requires_grad=True)

        # x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(self.fc1(out[:,-1,:]))
        return out

if __name__ == "__main__":
    arg = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = 'signal_pipeline_checkpoint.pth'

    n_epochs = 0
    toggle = 0

    pipeline = IMU_PIPELINE()
    utils = Helpers()

    optimizer = optim.Adam(pipeline.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    print(pipeline)
    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    _, _, imu_training, imu_testing, training_target, testing_target = utils.load_datasets()
    print(imu_training[0], training_target[0], imu_training[1442], training_target[1442])

    os.chdir(pipeline.var.root)
    imu_training_feat = np.copy(imu_training)
    imu_testing_feat = np.copy(imu_testing)
    imu_training_feat = utils.standarization(imu_training_feat)
    imu_testing_feat = utils.standarization(imu_testing_feat)

    for epoch in tqdm(range(n_epochs), desc="epochs"):
        Dataset = FINAL_DATASET(imu_training_feat, training_target)
        trainset, testset = random_split(Dataset, int(len(Dataset)*0.8), int(len(Dataset)*0.2))
        trainLoader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=4)
        tqdm_trainLoader = tqdm(trainLoader)
        # testDataset = FINAL_DATASET(imu_testing_feat, testing_target)
        testLoader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=4)
        tqdm_testLoader = tqdm(testLoader)

        num_samples = 0
        total_loss, total_correct, total_accuracy = 0.0, 0.0, 0.0
        pipeline.train()
        for batch_index, (feat, labels) in enumerate(tqdm_trainLoader):
            num_samples += feat.size(0)
            labels = labels[:,0,:]
            pred = pipeline(feat.float()).to(device)
            loss = criterion(pred*1000.0, (labels*1000.0).float())
            total_loss += loss.item()
            total_correct += pipeline.get_num_correct(pred, labels.float())
            total_accuracy = total_correct / num_samples
            tqdm_trainLoader.set_description('training: ' + '_loss: {:.4} correct: {} accuracy: {:.3}'.format(
                total_loss / num_samples, total_correct, 100.0*total_accuracy))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 0 and 'del' in arg:
            # _ = os.system('mv runs new_backup')
            _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

        tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
        tb.add_scalar("Train Loss", total_loss / num_samples, epoch)
        tb.add_scalar("Training Correct", total_correct, epoch)
        tb.add_scalar("Train Accuracy", total_accuracy, epoch)

        pipeline.eval()
        with torch.no_grad():
            num_samples = 0
            total_loss, total_correct, total_accuracy = 0.0, 0.0, 0.0
            for batch_index, (feat, labels) in enumerate(tqdm_testLoader):
                num_samples += feat.size(0)
                labels = labels[:,0,:]
                pred = pipeline(feat.float()).to(device)
                loss = criterion(pred*1000.0, (labels*1000.0).float())
                total_loss += loss.item()
                total_correct += pipeline.get_num_correct(pred, labels.float())
                total_accuracy = total_correct / num_samples
                tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3}'.format(
                    total_loss / num_samples, total_correct, 100.0*total_accuracy))

        tb.add_scalar("Testing Loss", total_loss / num_samples, epoch)
        tb.add_scalar("Testing Correct", total_correct, epoch)
        tb.add_scalar("Testing Accuracy", total_accuracy, epoch)
        tb.close()

        if epoch % 5 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': pipeline.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, pipeline.var.root + model_checkpoint)
            print('Model saved')
