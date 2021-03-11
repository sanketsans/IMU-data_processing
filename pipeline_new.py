import sys, os
import numpy as np
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
from tqdm import tqdm
sys.path.append('../')
from pipeline_encoders import IMU_ENCODER, TEMP_ENCODER, VIS_ENCODER
from helpers import Helpers
from FlowNetPytorch.models import FlowNetS
from variables import RootVariables
from torch.utils.tensorboard import SummaryWriter

class FusionPipeline(nn.Module):
    def __init__(self, checkpoint, test_folder, device=None):
        super(FusionPipeline, self).__init__()
        torch.manual_seed(2)
        self.device = device
        self.var = RootVariables()
        self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()
        self.temporalSeq = 16
        self.temporalSize = 8
        self.trim_frame_size = 150
        self.imuCheckpoint_file = 'signal_checkpoint0_' + test_folder[5:] + '.pth'
        self.frameCheckpoint_file = 'vision_checkpoint_' + test_folder[5:] +'.pth'

        ## IMU Models
        self.imuModel = IMU_ENCODER()
        imuCheckpoint = torch.load(self.var.root + 'datasets/' + test_folder[5:] + '/' + self.imuCheckpoint_file)
        self.imuModel.load_state_dict(imuCheckpoint['model_state_dict'])
        for params in self.imuModel.parameters():
            params.requires_grad = True

        ## FRAME MODELS
        self.frameModel =  VIS_ENCODER(self.checkpoint_path, self.device)
        frameCheckpoint = torch.load(self.var.root + 'datasets/' + test_folder[5:] + '/' + self.frameCheckpoint_file)
        self.frameModel.load_state_dict(frameCheckpoint['model_state_dict'])
        for params in self.frameModel.parameters():
            params.requires_grad = True

        ## TEMPORAL MODELS
        self.temporalModel = TEMP_ENCODER(self.temporalSize, self.device)

        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.dropuot = nn.Dropout(0.45)

        ##OTHER
        self.imu_encoder_params = None
        self.frame_encoder_params = None

        self.tensorboard_folder = 'pipeline_Adam' #'batch_64_Signal_outputs/'

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        self.imu_encoder_params = self.imuModel(imu_BatchData.float()).to(self.device)
        self.frame_encoder_params = self.frameModel(frame_BatchData.float()).to(self.device)

        return self.imu_encoder_params, self.frame_encoder_params

    def fusion_network(self, imu_params, frame_params):
        return torch.cat((frame_params, imu_params), dim=1).to(self.device)

    def temporal_modelling(self, fused_params):
        newParams = fused_params.reshape(fused_params.shape[0], self.temporalSeq, self.temporalSize)
        tempOut = self.temporalModel(newParams.float()).to(self.device)
        gaze_pred = self.activation(self.fc1(tempOut)).to(self.device)

        return gaze_pred

    def forward(self, batch_frame_data, batch_imu_data):
        imu_params, frame_params = self.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = self.fusion_network(imu_params, frame_params)
        coordinate = self.temporal_modelling(fused)

        return coordinate

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def get_original_coordinates(self, pred, labels):
        pred[:,0] *= 3.75*1920.0
        pred[:,1] *= 2.8125*1080.0

        labels[:,0] *= 3.75*1920.0
        labels[:,1] *= 2.8125*1080.0

        return pred, labels

class FINAL_DATASET(Dataset):
    def __init__(self, folder_type, imu_feat, labels):
        self.var = RootVariables()
        self.folder_type = folder_type
        self.labels = labels
        self.imu = imu_feat
        self.indexes = []
        checkedLast = False
        for index in range(len(self.labels)):
            check = np.isnan(self.labels[index])
            imu_check = np.isnan(self.imu[index])
            if check.any() or imu_check.any():
                continue
            else:
                self.indexes.append(index)

        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.indexes) # len(self.labels)

    def __getitem__(self, index):
        index = self.indexes[index]

        img =  np.load(self.var.root + self.folder_type + '/frames_' + str(index) +'.npy')
        targets = self.labels[index]
        targets[:,0] *= 0.2667
        targets[:,1] *= 0.3556

        return self.transforms(img).to(self.device), torch.from_numpy(self.imu[index]).to(self.device), torch.from_numpy(targets).to(self.device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_folder = 'test_CoffeeVendingMachine_S1'
    model_checkpoint = 'pipeline_checkpoint_' + test_folder[5:] + '.pth'
    flownet_checkpoint = 'flownets_EPE1.951.pth.tar'
    trim_frame_size = 150
    arg = 'del'
    n_epochs = 0
    # current_loss_mean_train, current_loss_mean_val, current_loss_mean_test = 0.0, 0.0,  0.0
    pipeline = FusionPipeline(flownet_checkpoint, test_folder, device)
    optimizer = optim.SGD(pipeline.parameters(), lr=1e-4, momentum=0.9)
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=-1)
    criterion = nn.L1Loss()
    print(pipeline)
    best_test_loss = 1000.0
    if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_test_loss = checkpoint['best_test_loss']
        # pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    utils = Helpers(test_folder, reset_dataset=1)
    imu_training, imu_testing, training_target, testing_target = utils.load_datasets()

    os.chdir(pipeline.var.root)

    for epoch in tqdm(range(n_epochs), desc="epochs"):
        if epoch > 0:
            utils = Helpers(test_folder, reset_dataset=0)
            imu_training, imu_testing, training_target, testing_target = utils.load_datasets()

        trainDataset = FINAL_DATASET('training_images', imu_training, training_target)
        trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
        tqdm_trainLoader = tqdm(trainLoader)
        testDataset = FINAL_DATASET('testing_images', imu_testing,  testing_target)
        testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
        tqdm_testLoader = tqdm(testLoader)

        if epoch == 0 and 'del' in arg:
            # _ = os.system('mv runs new_backup')
            _ = os.system('rm -rf ' + pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)

        num_samples = 0
        total_loss, total_correct, total_accuracy = [], 0.0, 0.0
        pipeline.train()
        tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
        for batch_index, (frame_feat, imu_feat, labels) in enumerate(tqdm_trainLoader):
            num_samples += frame_feat.size(0)
            labels = labels[:,0,:]
            pred = pipeline(frame_feat, imu_feat).to(device)

            loss = criterion(pred, labels.float())
            optimizer.zero_grad()
            loss.backward()
            ## add gradient clipping
            nn.utils.clip_grad_value_(pipeline.parameters(), clip_value=1.0)
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

        scheduler.step()
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
            for batch_index, (frame_feat, imu_feat, labels) in enumerate(tqdm_testLoader):
                num_samples += frame_feat.size(0)
                labels = labels[:,0,:]
                dummy_pts = (torch.ones(8, 2) * 0.5).to(device)
                dummy_pts[:,0] *= 1920
                dummy_pts[:,1] *= 1080

                pred = pipeline(frame_feat, imu_feat).to(device)
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

    # optimizer = optim.Adam([
    #                         {'params': imuModel.parameters(), 'lr': 1e-4},
    #                         {'params': frameModel.parameters(), 'lr': 1e-4},
    #                         {'params': temporalModel.parameters(), 'lr': 1e-4}
    #                         ], lr=1e-3)
