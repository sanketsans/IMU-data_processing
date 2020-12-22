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
from gaze_plotter import GET_DATAFRAME_FILES
from getDataset import FRAME_IMU_DATASET
from variables import RootVariables
from model_params import efficientPipeline

class FusionPipeline(nn.Module):
    def __init__(self, vars, args, checkpoint):
        super(FusionPipeline, self).__init__()
        torch.manual_seed(2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var = vars
        self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()
        self.temporalSeq = 18
        self.temporalSize = 64

        ## IMU Models
        self.imuModel = IMU_ENCODER(self.var.input_size, self.var.hidden_size, self.var.num_layers, self.var.num_classes, self.device)

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
        self.frame_imu_dataset = None
        self.frame_imu_dataLoader = None
        self.imu_encoder_params = None
        self.frame_encoder_params = None
        self.fused_params = None

    def get_dataset_dataloader(self, folder, trim_frame_size=150):
        self.rootfolder = folder
        os.chdir(self.var.root + self.rootfolder)

        self.frame_imu_dataset = FRAME_IMU_DATASET(self.var.root, self.rootfolder, trim_frame_size, device=self.device)
        # dataset_size = len(frame_imu_dataset)
        # indices = list(range(dataset_size))
        # valSplit_size = int(np.floor(validation_split * dataset_size))
        #
        # train_indices, val_indices = indices[valSplit_size:], indices[:valSplit_size]

        # Creating PT data samplers and loaders:
        # train_sampler = SequentialSampler(train_indices)
        # valid_sampler = SequentialSampler(val_indices)
        # trainLoader =  torch.utils.data.DataLoader(frame_imu_dataset, batch_size=self.var.batch_size, sampler=train_sampler)
        # valLoader = torch.utils.data.DataLoader(frame_imu_dataset, batch_size=self.var.batch_size, sampler=valid_sampler)
        # torch.save(frame_imu_dataset.stack_frames, self.var.root + self.rootfolder + 'stack_frames.pt')
        self.frame_imu_dataLoader = torch.utils.data.DataLoader(self.frame_imu_dataset, batch_size=self.var.batch_size, drop_last=True)

        return self.frame_imu_dataLoader
        # return trainLoader, valLoader

    def init_stage(self):
        # IMU Model
        self.imuModel_h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        self.imuModel_c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        # Temp Model
        self.tempModel_h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        self.tempModel_c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        self.imu_encoder_params, (h0, c0) = self.imuModel(imu_BatchData.float(), (self.imuModel_h0, self.imuModel_c0))
        # imu_encoder_params, _ = self.imuModel(imu_BatchData.float())
        self.frame_encoder_params = self.frameModel(frame_BatchData).to(self.device)
        self.imuModel_h0, self.imuModel_c0 = h0.detach(), c0.detach()
        # return imu_encoder_params
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
        # print(newParams.shape, gaze_pred.shape)
        self.tempModel_h0, self.tempModel_c0 = h0.detach(), c0.detach()

        return gaze_pred

    def forward(self, folder, batch_imu_data, batch_frame_data):
        # frame_imu_trainLoader = self.get_dataset_dataloader(folder)
        imu_params, frame_params = pipeline.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = pipeline.get_fusion_params(imu_params, frame_params)
        coordinate = pipeline.temporal_modelling(fused)

        return coordinate.type(torch.float32)


if __name__ == "__main__":
    # folder = 'imu_BookShelf_S1/'
    # if Path('train_loss.txt').is_file():
    #     os.system('rm train_loss.txt')
    #     os.system('rm val_loss.txt')
    #     os.system('rm test_loss.txt')
    # else:
    #     os.system('touch train_loss.txt')
    #     os.system('touch val_loss.txt')
    #     os.system('touch test_loss.txt')

    var = RootVariables()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    checkpoint = 'FlowNet2-S_checkpoint.pth.tar'

    pipeline = FusionPipeline(var, args, checkpoint)
    current_loss_mean = 0.0
    model_config = efficientPipeline
    optimizer = optim.Adam(pipeline.parameters(), lr=0.001, weight_decay=0.00001)
    loss_fn = nn.SmoothL1Loss()
    # optimizer = eval(model_config.optimizer)(pipeline.parameters(),**model_config.optimizer_parm)
    # scheduler = eval(model_config.scheduler)(optimizer,**model_config.scheduler_parm)
    # loss_fn = eval(model_config.loss_fn)()
    n_epochs = 1

    optimizer.zero_grad()
    train_loss = []
    val_loss = []
    test_loss = []

    for epoch in range(n_epochs):
        for subDir in os.listdir(var.root):
            if 'imu_' in subDir:
                print(subDir)
                pipeline.train()
                folder = subDir
                trainLoader = pipeline.get_dataset_dataloader(folder)
                pipeline.init_stage()

                tqdm_trainLoader = tqdm(trainLoader)
                # tfile = open(var.root + 'train_loss.txt', 'a+')
                # tfile.write(subDir + '\n')
                for batch_index, (frame_data, gaze_data, imu_data) in enumerate(tqdm_trainLoader):

                    coordinate = pipeline(folder, imu_data, frame_data).to(device)

                    gaze_data = gaze_data.reshape(gaze_data.shape[0], gaze_data.shape[1], -1)
                    avg_gaze_data = torch.sum(gaze_data, 2)
                    avg_gaze_data = avg_gaze_data / 8.0

                    loss = loss_fn(coordinate, avg_gaze_data.type(torch.float32).detach())
                    current_loss_mean = (current_loss_mean * batch_index + loss) / (batch_index + 1)
                    # print('loss: {} , lr: {}'.format(current_loss_mean, optimizer.param_groups[0]['lr']))
                    tqdm_trainLoader.set_description('loss: {:.4} lr:{:.6}'.format(
                        current_loss_mean, optimizer.param_groups[0]['lr']))

                    tfile.write(str(loss.item()) + '\n')

                    train_loss.append(loss.item())

                    loss.backward()
                    optimizer.step()
                    # scheduler.step(batch_index)
        #
        #         pipeline.eval()
        #         with torch.no_grad():
        #             folder = 'val_BookShelf_S1'
        #             vfile = open(var.root + 'val_loss.txt', 'a')
        #             vfile.write(folder + '\n')
        #             valLoader = pipeline.get_dataset_dataloader(folder)
        #             tqdm_valLoader = tqdm(valLoader)
        #             current_loss_mean_val = 0.0
        #             for batch_index, (frame_data, gaze_data, imu_data) in enumerate(tqdm_valLoader):
        #                 coordinate = pipeline(folder, imu_data, frame_data).to(device)
        #
        #                 gaze_data = gaze_data.reshape(gaze_data.shape[0], gaze_data.shape[1], -1)
        #                 avg_gaze_data = torch.sum(gaze_data, 2)
        #                 avg_gaze_data = avg_gaze_data / 8.0
        #
        #                 loss = loss_fn(coordinate, avg_gaze_data)
        #                 current_loss_mean_val = (current_loss_mean_val * batch_index + loss) / (batch_index + 1)
        #                 tqdm_valLoader.set_description('loss: {:.4} lr:{:.6}'.format(
        #                     current_loss_mean_val, optimizer.param_groups[0]['lr']))
        #                 val_loss.append(loss.item())
        #                 vfile.write(str(loss.item()) + '\n')
        #             vfile.close()
        #
        # with torch.no_grad():
        #     folder = 'test_BookShelf_S1'
        #     ttfile = open(var.root + 'test_loss.txt', 'a')
        #     ttfile.write(folder + '\n')
        #     testLoader = pipeline.get_dataset_dataloader(folder)
        #     tqdm_testLoader = tqdm(testLoader)
        #     current_loss_mean_test = 0.0
        #     for batch_index, (frame_data, gaze_data, imu_data) in enumerate(tqdm_testLoader):
        #         coordinate = pipeline(folder, imu_data, frame_data).to(device)
        #
        #         gaze_data = gaze_data.reshape(gaze_data.shape[0], gaze_data.shape[1], -1)
        #         avg_gaze_data = torch.sum(gaze_data, 2)
        #         avg_gaze_data = avg_gaze_data / 8.0
        #
        #         loss = loss_fn(coordinate, avg_gaze_data)
        #         current_loss_mean_test = (current_loss_mean_test * batch_index + loss) / (batch_index + 1)
        #         tqdm_testLoader.set_description('loss: {:.4} lr:{:.6}'.format(
        #             current_loss_mean_test, optimizer.param_groups[0]['lr']))
        #         test_loss.append(loss.item())
        #         ttfile.write(str(loss.item()) + '\n')
        #
        #     ttfile.close()
