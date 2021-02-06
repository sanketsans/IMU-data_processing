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
from flownet2.networks import FlowNetS
from variables import RootVariables
from helpers import Helpers
from torch.utils.tensorboard import SummaryWriter

class VISION_PIPELINE(nn.Module):
    def __init__(self, args, checkpoint_path, device, trim_frame_size=150, input_channels=6, batch_norm=False):
        super(VISION_PIPELINE, self).__init__()
        self.var = RootVariables()
        torch.manual_seed(1)
        self.device = device
        self.net = FlowNetS.FlowNetS(args, input_channels, batch_norm)
        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.net = nn.Sequential(*list(self.net.children())[0:9]).to(self.device)
        for i in range(len(self.net) - 1):
            self.net[i][1] = nn.ReLU()

        self.fc1 = nn.Linear(1024*4*4, 4096).to(self.device)
        self.fc2 = nn.Linear(4096, 256).to(self.device)
        self.fc3 = nn.Linear(256, 2).to(self.device)
        self.dropout = nn.Dropout(0.45)
        self.activation = nn.Sigmoid()
        # self.net[8][1] = nn.ReLU(inplace=False)
        self.net[8] = self.net[8][0]

        for params in self.net.parameters():
            params.requires_grad = True

        self.tensorboard_folder = 'frmae' #'BLSTM_signal_outputs_sell1/'

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

    def forward(self, input_img):
        out = self.net(input_img)
        out = out.reshape(-1, 1024*4*4)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.activation(self.fc3(out))

        return out

class FINAL_DATASET(Dataset):
    def __init__(self, feat, labels):
        self.feat = feat
        self.label = labels
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.transforms(self.feat[index]).to(self.device), torch.from_numpy(self.label[index]).to(self.device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    model_checkpoint = 'vision_pipeline_checkpoint.pth'
    flownet_checkpoint = 'FlowNet2-S_checkpoint.pth.tar'

    arg = 'del'
    n_epochs = 0
    toggle = 0
    trim_frame_size = 150
    pipeline = VISION_PIPELINE(args, flownet_checkpoint, device)

    optimizer = optim.Adam(pipeline.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    utils = Helpers()

    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    frame_training_feat, frame_testing_feat, _, _, training_target, testing_target = utils.load_datasets()

    os.chdir(pipeline.var.root)

    for epoch in tqdm(range(n_epochs), desc="epochs"):
        trainDataset = FINAL_DATASET(frame_training_feat, training_target)
        trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=4)
        tqdm_trainLoader = tqdm(trainLoader)
        testDataset = FINAL_DATASET(frame_testing_feat, testing_target)
        testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=4)
        tqdm_testLoader = tqdm(testLoader)

        num_samples = 0
        total_loss, total_correct, total_accuracy = 0.0, 0.0, 0.0
        pipeline.train()
        for batch_index, (feat, labels) in enumerate(tqdm_trainLoader):
            num_samples += feat.size(0)
            labels = labels[:,0,:]
            pred = pipeline(feat.float()).to(device)
            loss = criterion(pred*1000.0, (labels*1000.0).float())
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

        tb = SummaryWriter('runs/' + pipeline.tensorboard_folder)
        tb.add_scalar("Train Loss", total_loss / num_samples, epoch)
        tb.add_scalar("Training Correct", total_correct, epoch)
        tb.add_scalar("Train Accuracy", total_accuracy, epoch)

        pipeline.eval()
        with torch.no_grad():
            num_samples = 0
            total_loss, total_correct, total_accuracy = 0.0, 0.0, 0.0
            for batch_index, (feat, labels) in enumerate(tqdm_testLoader):
                num_samples += feat.size(0)
                labels = labels[:,0,:]
                pred = pipeline(feat.float()).to(device)
                loss = criterion(pred*1000.0, (labels*1000.0).float())
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
