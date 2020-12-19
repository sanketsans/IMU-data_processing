import sys, os
import torch.nn as nn
import cv2
import torch
import argparse
from encoder_imu import IMU_ENCODER
from encoder_vis import VIS_ENCODER
from gaze_plotter import GET_DATAFRAME_FILES
from getDataset import IMUDataset, ImageDataset
from variables import RootVariables

class FusionPipeline:
    def __init__(self, vars, args, checkpoint):
        # self.rootfolder = rootfolder
        self.device = torch.device("cpu")
        self.var = vars
        self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()

        ## IMU Models
        self.imuModel = IMU_ENCODER(self.var.input_size, self.var.hidden_size, self.var.num_layers, self.var.num_classes).to(self.device)

        ## FRAME MODELS
        self.args = args
        self.frameModel =  VIS_ENCODER(self.args, self.checkpoint_path, self.device)

    def get_dataset_dataloader(self, folder):
        self.rootfolder = folder
        os.chdir(self.var.root + self.rootfolder)

        frame_dataset = ImageDataset(self.var.root, self.rootfolder, device=self.device)
        frame_dataset.populate_data(frame_dataset.first_frame)
        # torch.save(self.frame_dataset.stack_frames, self.var.root + self.rootfolder + 'stack_frames.pt')
        frame_trainLoader = torch.utils.data.DataLoader(frame_dataset, batch_size=self.var.batch_size)

        self.frame_count = int(frame_dataset.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        get_dataframes = GET_DATAFRAME_FILES(self.frame_count)
        imu_dataset = IMUDataset(self.var.root, self.rootfolder)
        imu_trainLoader = torch.utils.data.DataLoader(imu_dataset, batch_size=self.var.batch_size)

        return imu_trainLoader, frame_trainLoader

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        imu_encoder_params = self.imuModel(imu_BatchData.float())
        frame_encoder_params = self.frameModel.run_model(frame_BatchData)
        # return imu_encoder_params
        return imu_encoder_params, frame_encoder_params

    def get_fusion_params(self, imu_params, frame_params):
        imu_activated = self.activation(imu_params)
        frames_activated = self.activation(frame_params)

        newIMU = imu_params * imu_activated
        newFrames = frame_params * frames_activated

        fused_params = torch.cat((newIMU, newFrames), dim=1)
        return fused_params

    def temporal_modelling(self, fused_params):
        pass
        # self.temporalModel =



if __name__ == "__main__":
    # folder = 'imu_BookShelf_S1/'
    var = RootVariables()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    checkpoint = 'FlowNet2-S_checkpoint.pth.tar'
    pipeline = FusionPipeline(var, args, checkpoint)
    for subDir in os.listdir(var.root):
        if 'imu_' in subDir:
            print(subDir)
            folder = subDir
            imu_trainLoader, frame_trainLoader = pipeline.get_dataset_dataloader(folder)
            a = iter(imu_trainLoader)
            imuData = next(a)
            b = iter(frame_trainLoader)
            frameData = next(b)
            imu_params, frame_params = pipeline.get_encoder_params(imuData, frameData)
            print(imu_params.shape, frame_params.shape)
            fused = pipeline.get_fusion_params(imu_params, frame_params)
            print(fused.shape)
