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
from encoder_imu import IMU_ENCODER, TEMP_ENCODER
from encoder_vis import VIS_ENCODER
from helpers import Helpers
# from getDataset import FRAME_IMU_DATASET
from variables import RootVariables
from torch.utils.tensorboard import SummaryWriter

class FusionPipeline(nn.Module):
    def __init__(self, args, checkpoint, trim_frame_size=150, device=None):
        super(FusionPipeline, self).__init__()
        torch.manual_seed(2)
        self.device = device
        self.var = RootVariables()
        self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()
        self.temporalSeq = 8
        self.temporalSize = 8
        self.trim_frame_size = trim_frame_size
        self.imuCheckpoint_file = 'signal_pipeline_checkpoint.pth'
        self.frameCheckpoint_file = 'vision_pipeline_checkpoint.pth'

        ## IMU Models
        self.imuModel = IMU_ENCODER(self.var.imu_input_size, self.device)
        # imuCheckpoint = torch.load(self.var.root + self.imuCheckpoint_file)
        # self.imuModel.load_state_dict(imuCheckpoint['model_state_dict'])
        for params in self.imuModel.parameters():
            params.requires_grad = True
        # self.imuRegressor = nn.Linear(self.var.hidden_size*2, 2).to(self.device)

        ## FRAME MODELS
        self.args = args
        self.frameModel =  VIS_ENCODER(self.args, self.checkpoint_path, self.device)
        # frameCheckpoint = torch.load(self.var.root + self.frameCheckpoint_file)
        # self.frameModel.load_state_dict(frameCheckpoint['model_state_dict'])
        for params in self.frameModel.parameters():
            params.requires_grad = True
        # self.frameRegressor = nn.Linear(256, 2).to(self.device)

        ## TEMPORAL MODELS
        self.temporalModel = TEMP_ENCODER(self.temporalSize, self.device)

        self.downsample_fc = nn.Linear(256, self.var.hidden_size*2).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.dropuot = nn.Dropout(0.45)
        # self.fc2 = nn.Linear(128, 2).to(self.device)
        # self.fusionLayer_sv = nn.Linear(512, 256).to(self.device)
        # self.fusionLayer_si = nn.Linear(512, 256).to(self.device)
        # self.finalFusion = nn.Linear(512, 256).to(self.device)

        ##OTHER
        self.imu_encoder_params = None
        self.frame_encoder_params = None

        self.tensorboard_folder = 'pipeline_SGD' #'batch_64_Signal_outputs/'

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        self.imu_encoder_params = self.imuModel(imu_BatchData.float()).to(self.device)
        self.frame_encoder_params = self.frameModel(frame_BatchData.float()).to(self.device)
        #self.imuModel_h0, self.imuModel_c0 = h0.detach(), c0.detach()

        return self.imu_encoder_params, self.frame_encoder_params

    def fusion_network(self, imu_params, frame_params):
        downsample_frame_params = F.relu(self.downsample_fc(frame_params)).to(self.device)
        return torch.cat((downsample_frame_params, imu_params), dim=1).to(self.device)

        # sv = self.activation(self.fusionLayer_sv(torch.cat((frame_params, imu_params), dim=1))).to(self.device)
        # si = self.activation(self.fusionLayer_si(torch.cat((frame_params, imu_params), dim=1))).to(self.device)
        #
        # newIMU = imu_params * sv
        # newFrames = frame_params * si
        # return F.relu(self.finalFusion(torch.cat((newFrames, newIMU), dim=1))).to(self.device)

    def temporal_modelling(self, fused_params):
        # self.fused_params = fused_params.unsqueeze(dim = 1)
        newParams = fused_params.reshape(fused_params.shape[0], self.temporalSeq, self.temporalSize)
        tempOut = self.temporalModel(newParams.float()).to(self.device)
        gaze_pred = self.activation(self.fc1(tempOut)).to(self.device)
        # regOut_1 = F.relu(self.fc1(tempOut)).to(self.device)
        # gaze_pred = self.activation(self.droput(self.fc2(regOut_1))).to(self.device)

        #self.tempModel_h0, self.tempModel_c0 = h0.detach(), c0.detach()

        return gaze_pred

    def forward(self, batch_frame_data, batch_imu_data):
        imu_params, frame_params = self.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = self.fusion_network(imu_params, frame_params)
        coordinate = self.temporal_modelling(fused)

        return coordinate

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

class FINAL_DATASET(Dataset):
    def __init__(self, frame_feat, imu_feat, labels):
        self.frame_feat = frame_feat
        self.imu_feat = imu_feat
        self.label = labels
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.transforms(self.frame_feat[index]).to(self.device), torch.from_numpy(self.imu_feat[index]).to(self.device), torch.from_numpy(self.label[index]).to(self.device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    test_folder = 'train_BookShelf_S1'
    model_checkpoint = 'pipeline_checkpoint_' + test_folder[6:] + '.pth'
    flownet_checkpoint = 'FlowNet2-S_checkpoint.pth.tar'
    trim_frame_size = 150
    arg = 'del'
    n_epochs = 0
    # current_loss_mean_train, current_loss_mean_val, current_loss_mean_test = 0.0, 0.0,  0.0
    pipeline = FusionPipeline(args, flownet_checkpoint, trim_frame_size, device)
    optimizer = optim.SGD(pipeline.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.L1Loss()
    print(pipeline)
    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    utils = Helpers(test_folder)
    frame_training_feat, frame_testing_feat, imu_training, imu_testing, training_target, testing_target = utils.load_datasets()
    imu_training_feat[:, :, 1] += 9.80665

    os.chdir(pipeline.var.root)
    imu_training_feat = np.copy(imu_training)
    imu_testing_feat = np.copy(imu_testing)
    imu_training_feat[:, :, 1] += 9.80665
    imu_training_feat = utils.standarization(imu_training_feat)
    imu_testing_feat = utils.standarization(imu_testing_feat)

    for epoch in tqdm(range(n_epochs), desc="epochs"):
        trainDataset = FINAL_DATASET(frame_training_feat, imu_training_feat, training_target)
        trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=4)
        tqdm_trainLoader = tqdm(trainLoader)
        testDataset = FINAL_DATASET(frame_testing_feat, imu_testing_feat, testing_target)
        testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=4)
        tqdm_testLoader = tqdm(testLoader)

        num_samples = 0
        total_loss, total_correct, total_accuracy = 0.0, 0.0, 0.0
        pipeline.train()
        for batch_index, (frame_feat, imu_feat, labels) in enumerate(tqdm_trainLoader):
            num_samples += frame_feat.size(0)
            labels = labels[:,0,:]
            pred = pipeline(frame_feat, imu_feat).to(device)
            imuPred = pipeline.activation(pipeline.imuRegressor(pipeline.imuModel(imu_feat.float()))).to(device)
            framePred = pipeline.activation(pipeline.frameRegressor(pipeline.frameModel(frame_feat.float()))).to(device)
            combLoss = criterion(pred, labels.float())
            imuLoss = criterion(imuPred, labels.float())
            frameLoss = criterion(framePred, labels.float())
            loss = combLoss + imuLoss + frameLoss
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

        tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
        tb.add_scalar("Train Loss", total_loss / num_samples, epoch)
        tb.add_scalar("Training Correct", total_correct, epoch)
        tb.add_scalar("Train Accuracy", total_accuracy, epoch)

        pipeline.eval()
        with torch.no_grad():
            num_samples = 0
            total_loss, total_correct, total_accuracy = 0.0, 0.0, 0.0
            for batch_index, (frame_feat, imu_feat, labels) in enumerate(tqdm_testLoader):
                num_samples += frame_feat.size(0)
                labels = labels[:,0,:]
                pred = pipeline(frame_feat, imu_feat).to(device)
                imuPred = pipeline.activation(pipeline.imuRegressor(pipeline.imuModel(imu_feat.float()))).to(device)
                framePred = pipeline.activation(pipeline.frameRegressor(pipeline.frameModel(frame_feat.float()))).to(device)
                combLoss = criterion(pred, labels.float())
                imuLoss = criterion(imuPred, labels.float())
                frameLoss = criterion(framePred, labels.float())
                loss = combLoss + imuLoss + frameLoss
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

    # optimizer = optim.Adam([
    #                         {'params': imuModel.parameters(), 'lr': 1e-4},
    #                         {'params': frameModel.parameters(), 'lr': 1e-4},
    #                         {'params': temporalModel.parameters(), 'lr': 1e-4}
    #                         ], lr=1e-3)
