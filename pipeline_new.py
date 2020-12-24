import sys, os
import numpy as np
import torch.nn as nn
import cv2
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SequentialSampler
import argparse
from tqdm import tqdm
from encoder_imu import IMU_ENCODER
from encoder_vis import VIS_ENCODER
from prepare_dataset import IMU_GAZE_DATASET
# from getDataset import FRAME_IMU_DATASET
from variables import RootVariables
# from model_params import efficientPipeline

class FusionPipeline(nn.Module):
    def __init__(self, args, checkpoint, trim_frame_size=150, device=None):
        super(FusionPipeline, self).__init__()
        torch.manual_seed(2)
        self.device = device
        self.var = RootVariables()
        self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()
        self.temporalSeq = 64
        self.temporalSize = 20
        self.trim_frame_size = trim_frame_size

        ## IMU Models
        self.imuModel = IMU_ENCODER(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, self.var.num_classes, self.device)

        ## FRAME MODELS
        self.args = args
        self.frameModel =  VIS_ENCODER(self.args, self.checkpoint_path, self.device)

        ## TEMPORAL MODELS
        self.temporalModel = IMU_ENCODER(self.temporalSize, self.var.hidden_size, self.var.num_layers, self.var.num_classes*8, self.device)

        self.fc1 = nn.Linear(self.var.num_classes*8, 512).to(self.device)
        self.fc2 = nn.Linear(512, 128).to(self.device)
        self.fc3 = nn.Linear(128, 2).to(self.device)
        # self.regressor = nn.Sequential(*[self.fc1, self.fc2, self.fc3])

        ##OTHER
        self.imu_gaze_dataset = None
        self.imu_gaze_dataloader = None
        self.frame_dataset = None
        self.frame_dataloader = None
        self.imu_encoder_params = None
        self.frame_encoder_params = None
        self.fused_params = None
        self.folder_iter = iter(os.listdir(self.var.root))
        self.folder_name = None

    def get_dataset_dataloader(self):
        self.imu_gaze_dataset = IMU_GAZE_DATASET(self.var.root, self.trim_frame_size, self.device)
        self.imu_gaze_dataloader = torch.utils.data.DataLoader(self.imu_gaze_dataset, batch_size=self.var.batch_size, drop_last=False)

        return self.imu_gaze_dataloader

    def init_stage(self):
        # IMU Model
        self.imuModel_h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        self.imuModel_c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        # Temp Model
        self.tempModel_h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        self.tempModel_c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        self.imu_encoder_params, (h0, c0) = self.imuModel(imu_BatchData.float(), (self.imuModel_h0, self.imuModel_c0))
        self.frame_encoder_params = self.frameModel(frame_BatchData).to(self.device)
        self.imuModel_h0, self.imuModel_c0 = h0.detach(), c0.detach()

        return self.imu_encoder_params, self.frame_encoder_params

    def get_fusion_params(self, imu_params, frame_params):
        newIMU = imu_params * self.activation(imu_params)
        newFrames = frame_params * self.activation(frame_params)

        self.fused_params = torch.cat((newIMU, newFrames), dim=1).to(self.device)
        return self.fused_params

    def temporal_modelling(self, fused_params):
        self.fused_params = fused_params.unsqueeze(dim = 1)
        newParams = fused_params.reshape(fused_params.shape[0], self.temporalSeq, self.temporalSize)
        # tempOut, _ = self.temporalModel(newParams.float())
        tempOut, (h0, c0) = self.temporalModel(newParams.float(), (self.tempModel_h0, self.tempModel_c0))
        regOut_1 = F.relu(self.fc1(tempOut)).to(self.device)
        regOut_2 = F.relu(self.fc2(regOut_1)).to(self.device)
        gaze_pred = self.activation(self.fc3(regOut_2)).to(self.device)

        self.tempModel_h0, self.tempModel_c0 = h0.detach(), c0.detach()

        return gaze_pred

    def get_frames_dataloader(self):
        while True:
            self.folder_name = next(self.folder_iter)
            if 'imu_' in self.folder_name:
                break

        self.frame_dataset = FRAME_DATASET(self.var.root, self.folder_name, self.trim_frame_size, self.device)
        # self.frame_dataloader = torch.utils.data.DataLoader(self.frame_dataset, batch_size=self.var.batch_size, drop_last=False)

        return self.frame_dataset, self.folder_name


    def forward(self, batch_imu_data, batch_frame_data):
        # frame_imu_trainLoader = self.get_dataset_dataloader(folder)
        imu_params, frame_params = pipeline.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = pipeline.get_fusion_params(imu_params, frame_params)
        coordinate = pipeline.temporal_modelling(fused)

        return coordinate.type(torch.float32)


if __name__ == "__main__":
    var = RootVariables()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    checkpoint = 'FlowNet2-S_checkpoint.pth.tar'
    trim_frame_size = 150
    current_loss_mean, current_loss_mean_val, current_loss_mean_test = 0.0, 0.0,  0.0
    pipeline = FusionPipeline(args, checkpoint, trim_frame_size, device)   ## MODEL DEFINE KIYA IDHAR
    optimizer = optim.SGD(pipeline.parameters(), lr=0.001, weight_decay=0.00001)
    loss_fn = nn.MSELoss()
    n_epochs = 1
    optimizer.zero_grad()
    current_loss = 10000.0
    # running_loss = 0.0
    # train_loss = []
    # val_loss = []
    # test_loss = []
    # pipeline.init_stage()
    # trainLoader = pipeline.get_dataset_dataloader()
    # tqdm_trainLoader = tqdm(trainLoader)
    dataset = UNIFIED_FRAME_DATASET(var.root, trim_frame_size, device)
    # for index, data, in enumerate(trainLoader):
    #     frameLoader_length, folder_name = pipeline.get_frames_dataloader()
    #     print(folder_name, frameLoader_length)
    #     if index > 8:
    #         break
            # for i, data in enumerate(trainLoader):
            #     f = next(b)
            #     frames += f.shape[0]
            #     print(frames, f.shape[0])
            #     if ((frame_dataset.frame_count - frames) < f.shape[0]):
            #         print('need to change folder', frames)
            #         break
            #     i , g = data
                # print(i.shape, g.shape, f.shape)

            # coordinate = pipeline(i, f).to(device)
            # print(coordinate, coordinate.shape)

            # for batch_index, (imu_data, gaze_data) in enumerate(tqdm(tqdm_trainLoader)):
            #     a = iter(frame_dataloader)
            #     stacked_frame_data = next(a)
            #
            #     print(imu_data.shape, gaze_data.shape, stacked_frame_data.shape)
            #
            #     coordinate = pipeline(imu_data, stacked_frame_data).to(device)
            # if folders_num > 2:
            #     break



            # for batch_index, (frame_data, gaze_data, imu_data) in enumerate(tqdm_trainLoader):
            #     imu_data = imu_data.reshape(imu_data.shape[0], imu_data.shape[2], -1)
            #     coordinate = pipeline(subDir, imu_data, frame_data).to(device)
            #
            #     gaze_data = gaze_data.view(gaze_data.shape[0], gaze_data.shape[2]*2, -1) ## *2 because 4 imu data per frame
            #     avg_gaze_data = torch.sum(gaze_data, 1)
            #     avg_gaze_data = avg_gaze_data / 8.0
            #     loss = loss_fn(coordinate, avg_gaze_data.type(torch.float32))
            #     # running_loss += loss.item() * imu_data.shape[0]
            #     current_loss_mean = (current_loss_mean * batch_index + loss) / (batch_index + 1)
            #     tqdm_trainLoader.set_description('loss: {:.4} lr:{:.6}'.format(
            #         current_loss_mean, optimizer.param_groups[0]['lr']))
            #
            #     # train_loss.append(loss.item())
            #     tfile.write(str(loss.item()) + '\n')
            #     loss.backward()
            #     optimizer.step()
            #
            # if (current_loss_mean < current_loss):
            #     current_loss = current_loss_mean
            #     print('SAVING MODEL')
            #     torch.save({
            #                 'epoch': epoch,
            #                 'model_state_dict': pipeline.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict(),
            #                 'loss': current_loss_mean
            #
