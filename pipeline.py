import sys, os
import torch.nn as nn
import cv2
import torch
import torch.nn.functional as F
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
        # self.rootfolder = rootfolder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var = vars
        self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()
        self.temporalSeq = 36
        self.temporalSize = 32

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

    def get_dataset_dataloader(self, folder):
        self.rootfolder = folder
        os.chdir(self.var.root + self.rootfolder)

        frame_imu_dataset = FRAME_IMU_DATASET(self.var.root, self.rootfolder, 500, device=self.device)
        # torch.save(frame_imu_dataset.stack_frames, self.var.root + self.rootfolder + 'stack_frames.pt')
        frame_imu_dataLoader = torch.utils.data.DataLoader(frame_imu_dataset, batch_size=self.var.batch_size)

        return frame_imu_dataLoader

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        imu_encoder_params = self.imuModel(imu_BatchData.float()).to(self.device)
        frame_encoder_params = self.frameModel(frame_BatchData).to(self.device)
        # return imu_encoder_params
        return imu_encoder_params, frame_encoder_params

    def get_fusion_params(self, imu_params, frame_params):
        imu_activated = self.activation(imu_params)
        frames_activated = self.activation(frame_params)

        newIMU = imu_params * imu_activated
        newFrames = frame_params * frames_activated

        fused_params = torch.cat((newIMU, newFrames), dim=1).to(self.device)
        return fused_params

    def temporal_modelling(self, fused_params):
        fused_params = fused_params.unsqueeze(dim = 1)
        newParams = fused_params.reshape(fused_params.shape[0], 36, 32)
        tempOut = self.temporalModel(newParams.float()).to(self.device)
        regOut_1 = F.relu(self.fc1(tempOut)).to(self.device)
        regOut_2 = F.relu(self.fc2(regOut_1)).to(self.device)
        gaze_pred = self.activation(self.fc3(regOut_2)).to(self.device)
        # print(newParams.shape, gaze_pred.shape)

        return gaze_pred

    def forward(self, folder, batch_imu_data, batch_frame_data):
        # frame_imu_trainLoader = self.get_dataset_dataloader(folder)
        imu_params, frame_params = pipeline.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = pipeline.get_fusion_params(imu_params, frame_params)
        coordinate = pipeline.temporal_modelling(fused)

        return coordinate


if __name__ == "__main__":
    # folder = 'imu_BookShelf_S1/'

    var = RootVariables()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    checkpoint = 'FlowNet2-S_checkpoint.pth.tar'

    pipeline = FusionPipeline(var, args, checkpoint)
    current_loss_mean = 0
    model_config = efficientPipeline
    optimizer = eval(model_config.optimizer)(pipeline.parameters(),**model_config.optimizer_parm)
    scheduler = eval(model_config.scheduler)(optimizer,**model_config.scheduler_parm)
    loss_fn = eval(model_config.loss_fn)()
    optimizer.zero_grad()
    for subDir in os.listdir(var.root):
        if 'imu_' in subDir:
            print(subDir)
            folder = subDir
            frame_imu_trainLoader = pipeline.get_dataset_dataloader(folder)
            a = iter(frame_imu_trainLoader)
            frame_data, gaze_data, imu_data = next(a)
            tqdm_trainLoader = tqdm(frame_imu_trainLoader)
            for batch_index, (frame_data, gaze_data, imu_data) in enumerate(tqdm_trainLoader):

                coordinate = pipeline(folder, imu_data, frame_data).to(device)

                gaze_data = gaze_data.reshape(gaze_data.shape[0], gaze_data.shape[1], -1)
                avg_gaze_data = torch.sum(gaze_data, 2)
                avg_gaze_data = avg_gaze_data / 8.0
                print(coordinate.shape, gaze_data.shape, frame_data.shape, imu_data.shape, avg_gaze_data.shape)

                loss = loss_fn(coordinate, avg_gaze_data)
                current_loss_mean = (current_loss_mean * batch_index + loss) / (batch_index + 1)
                # print('loss: {} , lr: {}'.format(current_loss_mean, optimizer.param_groups[0]['lr']))
                tqdm_trainLoader.set_description('loss: {:.4} lr:{:.6}'.format(
                    current_loss_mean, optimizer.param_groups[0]['lr']))
                scheduler.step(batch_index)
                print()

            break
