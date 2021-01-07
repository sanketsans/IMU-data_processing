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
import argparse
from tqdm import tqdm
from encoder_imu import IMU_ENCODER
from encoder_vis import VIS_ENCODER
from prepare_dataset import IMU_GAZE_FRAME_DATASET, UNIFIED_DATASET
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
        self.temporalSeq = 80
        self.temporalSize = 16
        self.trim_frame_size = trim_frame_size

        ## IMU Models
        self.imuModel = IMU_ENCODER(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, self.var.num_classes, self.device)

        ## FRAME MODELS
        self.args = args
        self.frameModel =  VIS_ENCODER(self.args, self.checkpoint_path, self.device)

        ## TEMPORAL MODELS
        self.temporalModel = IMU_ENCODER(self.temporalSize, self.var.hidden_size, self.var.num_layers, self.var.num_classes*4, self.device)

        self.fc1 = nn.Linear(self.var.num_classes*4, 512).to(self.device)
        self.droput = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 2).to(self.device)
        # self.regressor = nn.Sequential(*[self.fc1, self.fc2, self.fc3])

        ##OTHER
        self.unified_dataset = None
        self.imu_encoder_params = None
        self.frame_encoder_params = None
        self.fused_params = None

    def prepare_dataset(self):
        self.unified_dataset = IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.trim_frame_size)
        return self.unified_dataset

    def init_stage(self):
        # IMU Model
        self.imuModel_h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        self.imuModel_c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        # Temp Model
        self.tempModel_h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        self.tempModel_c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        self.imu_encoder_params = self.imuModel(imu_BatchData.float()).to(self.device)
        self.frame_encoder_params = self.frameModel(frame_BatchData.float()).to(self.device)
        #self.imuModel_h0, self.imuModel_c0 = h0.detach(), c0.detach()

        return self.imu_encoder_params, self.frame_encoder_params

    def get_fusion_params(self, imu_params, frame_params):
        newIMU = imu_params * self.activation(imu_params)
        newFrames = frame_params * self.activation(frame_params)

        self.fused_params = torch.cat((newIMU, newFrames), dim=1).to(self.device)
        return self.fused_params

    def temporal_modelling(self, fused_params):
        self.fused_params = fused_params.unsqueeze(dim = 1)
        newParams = fused_params.reshape(fused_params.shape[0], self.temporalSeq, self.temporalSize)
        tempOut = self.temporalModel(newParams.float()).to(self.device)
        regOut_1 = F.relu(self.fc1(tempOut)).to(self.device)
        gaze_pred = self.activation(self.droput(self.fc2(regOut_1))).to(self.device)

        #self.tempModel_h0, self.tempModel_c0 = h0.detach(), c0.detach()

        return gaze_pred

    def forward(self, batch_frame_data, batch_imu_data):
        imu_params, frame_params = self.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = self.get_fusion_params(imu_params, frame_params)
        coordinate = self.temporal_modelling(fused)

        return coordinate

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    model_checkpoint = 'pipeline_checkpoint.pth'
    flownet_checkpoint = 'FlowNet2-S_checkpoint.pth.tar'
    trim_frame_size = 150
    # current_loss_mean_train, current_loss_mean_val, current_loss_mean_test = 0.0, 0.0,  0.0
    pipeline = FusionPipeline(args, flownet_checkpoint, trim_frame_size, device)

    uni_dataset = pipeline.prepare_dataset()
    uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    uni_gaze_dataset = uni_dataset.gaze_datasets

    n_epochs = 0
    folders_num = 0
    start_index = 0
    current_loss = 1000.0
    # optimizer = optim.Adam([
    #                         {'params': imuModel.parameters(), 'lr': 1e-4},
    #                         {'params': frameModel.parameters(), 'lr': 1e-4},
    #                         {'params': temporalModel.parameters(), 'lr': 1e-4}
    #                         ], lr=1e-3)
    optimizer = optim.Adam(pipeline.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss()

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
            sliced_frame_dataset, sliced_imu_dataset, sliced_gaze_dataset = None, None, None
            train_loss, val_loss, test_loss = 0.0, 0.0, 0.0
            capture, frame_count = None, None
            #pipeline.init_stage()

            if 'imu_' in subDir:
                pipeline.train()
                # folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                end_index = start_index + frame_count - trim_frame_size*2
                sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
                sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                # sliced_frame_dataset = torch.load('framesExtracted_data_' + str(trim_frame_size) + '.pt')
                sliced_frame_dataset = np.load(str(pipeline.var.frame_size) + '_framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')

                unified_dataset = UNIFIED_DATASET(sliced_frame_dataset, sliced_imu_dataset, sliced_gaze_dataset, device)
                unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                tqdm_trainLoader = tqdm(unified_dataloader)
                for batch_index, (frame_data, imu_data, gaze_data) in enumerate(tqdm_trainLoader):
                    # frame_data = frame_data.permute(0, 3, 1, 2)
                    gaze_data = torch.sum(gaze_data, axis=1) / 4.0
                    coordinates = pipeline(frame_data, imu_data).to(device)
                    optimizer.zero_grad()
                    loss = loss_fn(coordinates, gaze_data.float())
                    train_loss += loss.item()
                    tqdm_trainLoader.set_description('loss: {:.4} lr:{:.6} lowest: {}'.format(
                        train_loss/(batch_index+1), optimizer.param_groups[0]['lr'], current_loss))
                    loss.backward()
                    optimizer.step()
                    # break

                if ((train_loss/len(unified_dataloader)) < current_loss):
                    current_loss = (train_loss/len(unified_dataloader))
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': pipeline.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': current_loss
                                }, pipeline.var.root + model_checkpoint)
                    print('Model saved')

                start_index = end_index
                with open(pipeline.var.root + 'train_loss.txt', 'a') as f:
                    f.write(str(train_loss/len(unified_dataloader)) + '\n')
                    f.close()

            if 'val_' in subDir:
                pipeline.eval()
                with torch.no_grad():
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(pipeline.var.root + subDir)
                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    end_index = start_index + frame_count - trim_frame_size*2
                    sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
                    sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                    sliced_frame_dataset = np.load(str(pipeline.var.frame_size) + '_framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')

                    unified_dataset = UNIFIED_DATASET(sliced_frame_dataset, sliced_imu_dataset, sliced_gaze_dataset, device)
                    unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                    tqdm_valLoader = tqdm(unified_dataloader)
                    for batch_index, (frame_data, imu_data, gaze_data) in enumerate(tqdm_valLoader):
                        gaze_data = torch.sum(gaze_data, axis=1) / float(gaze_data.shape[1])
                        coordinates = pipeline(frame_data, imu_data).to(device)
                        loss = loss_fn(coordinates, gaze_data.float())
                        val_loss += loss.item()
                        tqdm_valLoader.set_description('loss: {:.4} lr:{:.6}'.format(
                            val_loss/(batch_index+1), optimizer.param_groups[0]['lr']))

                    with open(pipeline.var.root + 'validation_loss.txt', 'a') as f:
                        f.write(str(val_loss/len(unified_dataloader)) + '\n')
                        f.close()

                start_index = end_index

            if 'test_' in subDir:
                pipeline.eval()
                with torch.no_grad():
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(pipeline.var.root + subDir)
                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    end_index = start_index + frame_count - trim_frame_size*2
                    sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
                    sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                    sliced_frame_dataset = np.load(str(pipeline.var.frame_size) + '_framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')

                    unified_dataset = UNIFIED_DATASET(sliced_frame_dataset, sliced_imu_dataset, sliced_gaze_dataset, device)
                    unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                    tqdm_testLoader = tqdm(unified_dataloader)
                    for batch_index, (frame_data, imu_data, gaze_data) in enumerate(tqdm_testLoader):
                        gaze_data = torch.sum(gaze_data, axis=1) / 4.0
                        coordinates = pipeline(frame_data, imu_data).to(device)
                        loss = loss_fn(coordinates, gaze_data.float())
                        test_loss += loss.item()
                        tqdm_testLoader.set_description('loss: {:.4} lr:{:.6}'.format(
                            test_loss/(batch_index+1), optimizer.param_groups[0]['lr']))

                    with open(pipeline.var.root + 'testing_loss.txt', 'a') as f:
                        f.write(str(test_loss/len(unified_dataloader)) + '\n')
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
