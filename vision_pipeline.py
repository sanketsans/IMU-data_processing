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
from torch.utils.tensorboard import SummaryWriter

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
        for i in range(len(self.net) - 1):
            self.net[i][1] = nn.ReLU()
        self.fc1 = nn.Linear(1024*4*4, 4096).to(self.device)
        self.fc2 = nn.Linear(4096, 256).to(self.device)
        self.fc3 = nn.Linear(256, 2).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()
        # self.net[8][1] = nn.ReLU(inplace=False)
        self.net[8] = self.net[8][0]

        for params in self.net.parameters():
            params.requires_grad = True

        self.loss_fn = nn.SmoothL1Loss()
        self.tensorboard_folder = 'b8_hidden_256_Vision_outputs/'
        self.total_loss, self.current_loss, self.total_accuracy, self.total_correct = 0.0, 10000.0, 0.0, 0
        self.uni_frame_dataset, self.uni_gaze_dataset = None, None
        self.sliced_frame_dataset, self.sliced_gaze_dataset = None, None
        self.unified_dataset = None
        self.start_index, self.end_index = 0, 0

    def prepare_dataset(self):
        self.unified_dataset = IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.trim_frame_size)
        return self.unified_dataset

    def get_num_correct(self, pred, label):
        return (torch.abs(pred - label) <= 30.0).all(axis=1).sum().item()

    def forward(self, input_img):
        out = self.net(input_img)
        out = out.reshape(-1, 1024*4*4)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.activation(self.fc3(out))

        return out*1000.0

    def engine(self, data_type='imu_', optimizer=None):
        self.total_loss, self.total_accuracy, self.total_correct = 0.0, 0.0, 0
        capture = cv2.VideoCapture('scenevideo.mp4')
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.end_index = self.start_index + frame_count - self.trim_frame_size*2

        self.sliced_frame_dataset = np.load(str(self.var.frame_size) + '_framesExtracted_data_' + str(self.trim_frame_size) + '.npy', mmap_mode='r')
        self.sliced_gaze_dataset = self.uni_gaze_dataset[self.start_index: self.end_index]
        self.unified_dataset = VISION_DATASET(self.sliced_frame_dataset, self.sliced_gaze_dataset, self.device)

        unified_dataloader = torch.utils.data.DataLoader(self.unified_dataset, batch_size=self.var.batch_size, num_workers=0, drop_last=True)
        tqdm_dataLoader = tqdm(unified_dataloader)
        for batch_index, (frame_data, gaze_data) in enumerate(tqdm_dataLoader):
            gaze_data = (torch.sum(gaze_data, axis=1) / 4.0)
            coordinates = self.forward(frame_data).to(device)
            loss = self.loss_fn(coordinates, gaze_data.float())
            self.total_loss += loss.item()
            self.total_correct += pipeline.get_num_correct(coordinates, gaze_data.float())
            self.total_accuracy = self.total_correct / (coordinates.size(0) * (batch_index+1))
            tqdm_dataLoader.set_description(data_type + '_loss: {:.4} accuracy: {:.3} lowest: {}'.format(
                self.total_loss, self.total_accuracy, self.current_loss))

            if 'imu_' in data_type:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.start_index = self.end_index

        return self.total_loss, self.total_accuracy

class VISION_DATASET(Dataset):
    def __init__(self, frame_dataset, gaze_dataset, device=None):
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.frame_data = frame_dataset
        self.gaze_data = gaze_dataset
        self.device = device

    def __len__(self):
        return len(self.gaze_data) -1

    def __getitem__(self, index):
        checkedLast = False
        while True:
            check = np.isnan(self.gaze_data[index])
            if check.any():
                index = (index - 1) if checkedLast else (index + 1)
                if index == self.__len__():
                    checkedLast = True
            else:
                break
        return self.transforms(self.frame_data[index]).to(self.device), torch.from_numpy(self.gaze_data[index]*1000.0).to(self.device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    model_checkpoint = 'vision_pipeline_checkpoint.pth'
    flownet_checkpoint = 'FlowNet2-S_checkpoint.pth.tar'

    arg = 'ag'
    n_epochs = 1
    trim_frame_size = 150
    pipeline = VISION_PIPELINE(args, flownet_checkpoint, device)
    uni_dataset = pipeline.prepare_dataset()
    pipeline.uni_gaze_dataset = uni_dataset.gaze_datasets

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
