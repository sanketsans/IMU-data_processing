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
from torch.utils.tensorboard import SummaryWriter

class FusionPipeline(nn.Module):
    def __init__(self, args, checkpoint, trim_frame_size=150, device=None):
        super(FusionPipeline, self).__init__()
        torch.manual_seed(2)
        self.device = device
        self.var = RootVariables()
        self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()
        self.temporalSeq = 32
        self.temporalSize = 8
        self.trim_frame_size = trim_frame_size
        self.imuCheckpoint_file = 'hidden_256_60e_signal_pipeline_checkpoint.pth'
        self.frameCheckpoint_file = 'hidden_256_55e_vision_pipeline_checkpoint.pth'

        ## IMU Models
        self.imuModel = IMU_ENCODER(self.var.imu_input_size, self.device)
        imuCheckpoint = torch.load(self.var.root + self.imuCheckpoint_file)
        self.imuModel.load_state_dict(imuCheckpoint['model_state_dict'])

        ## FRAME MODELS
        self.args = args
        self.frameModel =  VIS_ENCODER(self.args, self.checkpoint_path, self.device)
        frameCheckpoint = torch.load(self.var.root + self.frameCheckpoint_file)
        self.frameModel.load_state_dict(frameCheckpoint['model_state_dict'])

        ## TEMPORAL MODELS
        self.temporalModel = IMU_ENCODER(self.temporalSize, self.device)

        self.fc1 = nn.Linear(self.var.hidden_size*2, 128).to(self.device)
        self.droput = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2).to(self.device)
        self.fusionLayer_sv = nn.Linear(512, 256).to(self.device)
        self.fusionLayer_si = nn.Linear(512, 256).to(self.device)
        self.finalFusion = nn.Linear(512, 256).to(self.device)

        ##OTHER
        self.imu_encoder_params = None
        self.frame_encoder_params = None

        self.loss_fn = nn.SmoothL1Loss()
        self.tensorboard_folder = 'batch_64_Signal_outputs/'
        self.total_loss, self.current_loss = 0.0, 10000.0
        self.uni_imu_dataset, self.uni_gaze_dataset = None, None
        self.sliced_imu_dataset, self.sliced_gaze_dataset, self.sliced_frame_dataset = None, None, None
        self.unified_dataset = None
        self.start_index, self.end_index, self.total_accuracy = 0, 0, 0.0

    def prepare_dataset(self):
        return IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.trim_frame_size)

    # def get_sigmoid_mask(self, params):
    #     return self.activation(params)

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

    def fusion_network(self, imu_params, frame_params):
        print(frame_params.shape, imu_params.shape)
        sv = self.activation(self.fusionLayer_sv(torch.cat((frame_params, imu_params), dim=1))).to(self.device)
        si = self.activation(self.fusionLayer_si(torch.cat((frame_params, imu_params), dim=1))).to(self.device)

        newIMU = imu_params * sv
        newFrames = frame_params * si
        return F.relu(self.finalFusion(torch.cat((newFrames, newIMU), dim=1))).to(self.device)

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
        fused = self.fusion_network(imu_params, frame_params)
        coordinate = self.temporal_modelling(fused)

        return coordinate*1000.0

    def get_num_correct(self, pred, label):
        return (torch.abs(pred - label) < 30.0).all(axis=1).mean()

    def engine(self, data_type='imu_', optimizer=None):
        self.total_loss, self.total_accuracy = 0.0, 0.0
        capture = cv2.VideoCapture('scenevideo.mp4')
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.end_index = self.start_index + frame_count - self.trim_frame_size*2

        self.sliced_imu_dataset = self.uni_imu_dataset[self.start_index: self.end_index].detach().cpu().numpy()
        self.sliced_gaze_dataset = self.uni_gaze_dataset[self.start_index: self.end_index].detach().cpu().numpy()
        self.sliced_frame_dataset = np.load(str(self.var.frame_size) + '_framesExtracted_data_' + str(self.trim_frame_size) + '.npy', mmap_mode='r')

        self.unified_dataset = UNIFIED_DATASET(self.sliced_frame_dataset, self.sliced_imu_dataset, self.sliced_gaze_dataset, self.device)

        unified_dataloader = torch.utils.data.DataLoader(self.unified_dataset, batch_size=self.var.batch_size, num_workers=0, drop_last=True)
        tqdm_dataLoader = tqdm(unified_dataloader)
        for batch_index, (frame_data, imu_data, gaze_data) in enumerate(tqdm_dataLoader):
            gaze_data = (torch.sum(gaze_data, axis=1) / 4.0)
            coordinates = self.forward(frame_data, imu_data).to(self.device)
            print(coordinates)
            loss = self.loss_fn(coordinates, gaze_data.float())
            self.total_loss += loss.item()
            # total_correct += pipeline.get_num_correct(coordinates, gaze_data.float())
            # self.total_accuracy = total_correct / (coordinates.size(0) * (batch_index+1))
            tqdm_dataLoader.set_description(data_type + '_loss: {:.4} lowest: {}'.format(
                self.total_loss, self.current_loss))

            if 'imu_' in data_type:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.start_index = self.end_index

        return self.total_loss, self.total_accuracy



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
    pipeline.uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    pipeline.uni_gaze_dataset = uni_dataset.gaze_datasets

    arg = 'del'
    n_epochs = 1
    current_loss = 1000.0
    # optimizer = optim.Adam([
    #                         {'params': imuModel.parameters(), 'lr': 1e-4},
    #                         {'params': frameModel.parameters(), 'lr': 1e-4},
    #                         {'params': temporalModel.parameters(), 'lr': 1e-4}
    #                         ], lr=1e-3)
    optimizer = optim.Adam(pipeline.parameters(), lr=1e-4)

    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pipeline.current_loss = checkpoint['loss']
        print('Model loaded')


    for epoch in tqdm(range(n_epochs), desc="epochs"):
        pipeline.start_index = 0
        for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
            loss, accuracy = 0.0, 0.0
            #pipeline.init_stage()
            if 'imu_' in subDir:
                pipeline.train()
                # folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                if epoch == 0 and 'del' in arg:
                    _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                loss, accuracy = pipeline.engine('imu_', optimizer)

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
                pipeline.eval()
                with torch.no_grad():
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(pipeline.var.root + subDir)
                    if epoch == 0 and 'del' in arg:
                        _ = os.system('rm -rf runs/' + pipeline.tensorboard_folder)

                    loss, accuracy = pipeline.engine('val_')

                    tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
                    tb.add_scalar("Loss", loss, epoch)
                    tb.add_scalar("Accuracy", accuracy, epoch)
                    tb.close()

        if epoch % 5 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': pipeline.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': pipeline.current_loss
                        }, pipeline.var.root + model_checkpoint)
            print('Model saved')
