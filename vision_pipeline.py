import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
from tqdm import tqdm
sys.path.append('../')
from prepare_dataset import IMU_GAZE_FRAME_DATASET
from flownet2.networks import FlowNetS
from variables import RootVariables

class VISION_PIPELINE(nn.Module):
    def __init__(self, args, checkpoint_path, device, trim_frame_size=150, input_channels=6, batch_norm=False):
        super(VISION_PIPELINE, self).__init__()
        self.var = RootVariables()
        torch.manual_seed(1)
        self.device = device
        self.trim_frame_size = trim_frame_size
        self.net = FlowNetS.FlowNetS(args, input_channels, batch_norm)
        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.net = nn.Sequential(*list(self.net.children())[0:9]).to(self.device)
        # for i in range(len(self.net) - 1):
        #     self.net[i][1] = nn.ReLU()
        self.fc1 = nn.Linear(1024*4*4, 4096).to(self.device)
        self.fc2 = nn.Linear(4096, 1024).to(self.device)
        self.fc3 = nn.Linear(1024, 2).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()
        # self.net[8][1] = nn.ReLU(inplace=False)
        self.net[8] = self.net[8][0]

        for params in self.net.parameters():
            params.requires_grad = True

    def prepare_dataset(self):
        self.unified_dataset = IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.trim_frame_size)
        return self.unified_dataset

    def forward(self, input_img):
        print(input_img.shape)
        out = self.net(input_img)
        out = out.reshape(-1, 1024*4*4)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.activation(self.droput(self.fc3(out)))

        return out

class VISION_DATASET(Dataset):
    def __init__(self, frame_dataset, gaze_dataset, device=None):
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.frame_data = frame_dataset
        self.gaze_data = gaze_dataset
        self.device = device

    def __len__(self):
        return len(self.gaze_data) -1

    def __getitem__(self, index):
        return self.transforms(self.frame_data[index]).to(self.device), torch.from_numpy(self.gaze_data[index]).to(self.device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    model_checkpoint = 'vision_pipeline_checkpoint.pth'
    flownet_checkpoint = 'FlowNet2-S_checkpoint.pth.tar'

    n_epochs = 1
    folders_num = 0
    start_index = 0
    current_loss = 1000.0
    trim_frame_size = 150
    pipeline = VISION_PIPELINE(args, flownet_checkpoint, device)
    uni_dataset = pipeline.prepare_dataset()
    # uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    uni_gaze_dataset = uni_dataset.gaze_datasets
    optimizer = optim.Adam(pipeline.parameters(), lr=0.001)
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

            if 'imu_' in subDir:
                pipeline.train()
                # folders_num += 1
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                end_index = start_index + frame_count - trim_frame_size*2
                sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                # sliced_frame_dataset = torch.load('framesExtracted_data_' + str(trim_frame_size) + '.pt')
                sliced_frame_dataset = np.load(str(pipeline.var.frame_size) + '_framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')

                unified_dataset = VISION_DATASET(sliced_frame_dataset, sliced_gaze_dataset, device)
                unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                tqdm_trainLoader = tqdm(unified_dataloader)
                for batch_index, (frame_data, gaze_data) in enumerate(tqdm_trainLoader):
                    gaze_data = torch.sum(gaze_data, axis=1) / 4.0
                    coordinates = pipeline(frame_data).to(device)
                    loss = loss_fn(coordinates, gaze_data.float())
                    train_loss += loss.item()
                    tqdm_trainLoader.set_description('loss: {:.4} lr:{:.6} lowest: {}'.format(
                        loss.item(), optimizer.param_groups[0]['lr'], current_loss))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

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
                with open(pipeline.var.root + 'vision_train_loss.txt', 'a') as f:
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
                    sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                    sliced_frame_dataset = np.load(str(pipeline.var.frame_size) + '_framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')

                    unified_dataset = VISION_DATASET(sliced_frame_dataset, sliced_gaze_dataset, device)
                    unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                    tqdm_valLoader = tqdm(unified_dataloader)
                    for batch_index, (frame_data, gaze_data) in enumerate(tqdm_valLoader):
                        gaze_data = torch.sum(gaze_data, axis=1) / float(gaze_data.shape[1])
                        coordinates = pipeline(frame_data).to(device)
                        loss = loss_fn(coordinates, gaze_data.float())
                        val_loss += loss.item()
                        tqdm_valLoader.set_description('loss: {:.4} lr:{:.6}'.format(
                            loss.item(), optimizer.param_groups[0]['lr']))

                    with open(pipeline.var.root + 'vision_validation_loss.txt', 'a') as f:
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
                    sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                    sliced_frame_dataset = np.load(str(pipeline.var.frame_size) + '_framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')

                    unified_dataset = VISION_DATASET(sliced_frame_dataset, sliced_gaze_dataset, device)
                    unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                    tqdm_testLoader = tqdm(unified_dataloader)
                    for batch_index, (frame_data, gaze_data) in enumerate(tqdm_testLoader):
                        gaze_data = torch.sum(gaze_data, axis=1) / 4.0
                        coordinates = pipeline(frame_data).to(device)
                        loss = loss_fn(coordinates, gaze_data.float())
                        test_loss += loss.item()
                        tqdm_testLoader.set_description('loss: {:.4} lr:{:.6}'.format(
                            loss.item(), optimizer.param_groups[0]['lr']))

                    with open(pipeline.var.root + 'vision_testing_loss.txt', 'a') as f:
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
