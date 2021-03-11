import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
from tqdm import tqdm
sys.path.append('../')
from FlowNetPytorch.models import FlowNetS
# from flownet2.networks import FlowNetSD
from variables import RootVariables
from helpers import Helpers
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import rotate
import random


class VISION_PIPELINE(nn.Module):
    def __init__(self, checkpoint_path, device, trim_frame_size=150, input_channels=6, batch_norm=False):
        super(VISION_PIPELINE, self).__init__()
        self.var = RootVariables()
        torch.manual_seed(1)
        self.device = device
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
        # parser.add_argument("--rgb_max", type=float, default=255.)
        # args = parser.parse_args()
        self.net = FlowNetS.FlowNetS(batch_norm)

        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.net = nn.Sequential(*list(self.net.children())[0:10]).to(self.device)
        # for i in range(len(self.net) - 1):
        #     self.net[i][1] = nn.ReLU()

        self.fc1 = nn.Linear(1024*6*8, 4096).to(self.device)
        self.fc2 = nn.Linear(4096, 64).to(self.device)
        self.fc3 = nn.Linear(64, 2).to(self.device)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.Sigmoid()
        # self.net[8][1] = nn.ReLU(inplace=False)
        # self.net[8] = self.net[8][0]

        for params in self.net.parameters():
            params.requires_grad = True

        self.tensorboard_folder = 'vision_SGD' #'BLSTM_signal_outputs_sell1/'

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def forward(self, input_img):
        out = self.net(input_img)
        out = out.reshape(-1, 1024*6*8)
        out = F.leaky_relu(self.fc1(out), 0.1)
        out = F.leaky_relu(self.fc2(out), 0.1)
        out = self.fc3(out)

        return out

    def get_original_coordinates(self, pred, labels):
        # pred[:,0] *= 3.75*1920.0
        # pred[:,1] *= 2.8125*1080.0
        #
        # labels[:,0] *= 3.75*1920.0
        # labels[:,1] *= 2.8125*1080.0

        pred[:,0] *= 3.75
        pred[:,1] *= 2.8125

        labels[:,0] *= 3.75
        labels[:,1] *= 2.8125

        return pred, labels

class FINAL_DATASET(Dataset):
    def __init__(self, folder_type, labels):
        self.var = RootVariables()
        self.folder_type = folder_type
        self.labels = labels
        self.indexes = []
        checkedLast = False
        for index in range(len(self.labels)):
            check = np.isnan(self.labels[index])
            if check.any():
                continue
            else:
                self.indexes.append(index)


        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.indexes) # len(self.labels)

    def __getitem__(self, index):
        index = self.indexes[index]

        img =  np.load(self.var.root + self.folder_type + '/frames_' + str(index) +'.npy')
        targets = self.labels[index]
        #targets[:,0] *= 0.2667
        #targets[:,1] *= 0.3556

        targets[:,0] *= 512.0
        targets[:,1] *= 384.0

        return self.transforms(img).to(self.device), torch.from_numpy(targets).to(self.device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    var = RootVariables()
    # test_folder = 'test_InTheDeak_S2'
    lastFolder, newFolder = None, None
    for index, subDir in enumerate(sorted(os.listdir(var.root))):
        #if 'train_BookShelf' in subDir:
        #    continue
        if 'train_' in subDir:
            newFolder = subDir

            _ = os.system('mv ' + newFolder + ' test_' + newFolder[6:])
            test_folder = 'test_' + newFolder[6:]
            if lastFolder is not None:
                print('Last folder changed')
                _ = os.system('mv test_' + lastFolder[6:] + ' ' + lastFolder)

            print(newFolder, lastFolder)
            os.chdir(var.root)
            model_checkpoint = 'vision_checkpointAdam_' + test_folder[5:] + '.pth'
            flownet_checkpoint = 'flownets_EPE1.951.pth.tar'
            # flownet_checkpoint = 'FlowNet2-SD_checkpoint.pth.tar'

            arg = 'del'
            n_epochs = 1
            toggle = 0
            trim_frame_size = 150
            pipeline = VISION_PIPELINE(flownet_checkpoint, device)
#            print(pipeline)
            optimizer = optim.Adam(pipeline.parameters(), lr=1e-4) #, momentum=0.9)
            lambda1 = lambda epoch: 0.85 ** epoch
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            criterion = nn.L1Loss()
            best_test_loss = 1000.0
            if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
                checkpoint = torch.load(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
                pipeline.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                best_test_loss = checkpoint['best_test_loss']
                # pipeline.current_loss = checkpoint['loss']
                print('Model loaded')

            utils = Helpers(test_folder, reset_dataset=1)
            _, _, training_target, testing_target = utils.load_datasets()
            os.chdir(pipeline.var.root)

            for epoch in tqdm(range(n_epochs), desc="epochs"):
                if epoch > 0:
                    utils = Helpers(test_folder, reset_dataset=0)
                    _, _, training_target, testing_target = utils.load_datasets()
#                ttesting_target = np.copy(testing_target)
                trainDataset = FINAL_DATASET('training_images', training_target)
                trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
                testDataset = FINAL_DATASET('testing_images', testing_target)
                testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)

                tqdm_trainLoader = tqdm(trainLoader)
                tqdm_testLoader = tqdm(testLoader)

                if epoch == 0 and 'del' in arg:
                    # _ = os.system('mv runs new_backup')
                    _ = os.system('rm -rf ' + pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)

                num_samples = 0
                total_loss, total_correct, total_accuracy = [], 0.0, 0.0
                trainPD, testPD = [], []
                pipeline.train()
                tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
                for batch_index, (feat, labels) in enumerate(tqdm_trainLoader):
                    num_samples += feat.size(0)
                    labels = labels[:,0,:]
                    # labels[:,0] *= 0.2667
                    # labels[:,1] *= 0.3556
                    pred = pipeline(feat.float()).to(device)

                    loss = criterion(pred, labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    ## add gradient clipping
                    nn.utils.clip_grad_value_(pipeline.parameters(), clip_value=1.0)
                    optimizer.step()

                    with torch.no_grad():

                        pred, labels = pipeline.get_original_coordinates(pred, labels)

                        dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
                        if batch_index > 0:
                            trainPD = torch.cat((trainPD, dist), 1)
                        else:
                            trainPD = dist

                        total_loss.append(loss.detach().item())
                        total_correct += pipeline.get_num_correct(pred, labels.float())
                        total_accuracy = total_correct / num_samples
                        tqdm_trainLoader.set_description('training: ' + '_loss: {:.4} correct: {} accuracy: {:.3} MPD: {} lr:{}'.format(
                            np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(trainPD), optimizer.param_groups[0]['lr']))

  #                      if batch_index % 10 :
  #                          tb.add_scalar("Train Pixel Distance", torch.mean(trainPD[len(trainPD)-10:]), batch_index + (epoch*len(trainLoader)))

                scheduler.step()
                pipeline.eval()
                with torch.no_grad():
                    tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
                    tb.add_scalar("Train Loss", np.mean(total_loss), epoch)
                    #tb.add_scalar("Training Correct", total_correct, epoch)
                    tb.add_scalar("Train Accuracy", total_accuracy, epoch)
                    tb.add_scalar("Mean train pixel dist", torch.mean(trainPD), epoch)

                    num_samples = 0
                    total_loss, total_correct, total_accuracy = [], 0.0, 0.0
                    dummy_correct, dummy_accuracy = 0.0, 0.0
                    for batch_index, (feat, labels) in enumerate(tqdm_testLoader):
                        num_samples += feat.size(0)
                        labels = labels[:,0,:]
                        dummy_pts = (torch.ones(8, 2) * 0.5).to(device)
                        dummy_pts[:,0] *= 1920.0
                        dummy_pts[:,1] *= 1080.0
                        # labels[:,0] *= 0.2667
                        # labels[:,1] *= 0.3556

                        pred = pipeline(feat.float()).to(device)
                        loss = criterion(pred, labels.float())

                        pred, labels = pipeline.get_original_coordinates(pred, labels)

                        dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
                        if batch_index > 0:
                            testPD = torch.cat((testPD, dist), 0)
                        else:
                            testPD = dist
#                        print(pred, labels, dist)

                        total_loss.append(loss.detach().item())
                        total_correct += pipeline.get_num_correct(pred, labels.float())
                        dummy_correct += pipeline.get_num_correct(dummy_pts.float(), labels.float())
                        dummy_accuracy = dummy_correct / num_samples
                        total_accuracy = total_correct / num_samples
                        tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3} MPD: {} DAcc: {:.4}'.format(
                            np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(testPD), np.floor(100.0*dummy_accuracy)))

 #                       if batch_index % 10 :
 #                           tb.add_scalar("Test Pixel Distance", torch.mean(testPD[len(testPD)-10:]), batch_index+(epoch*len(testLoader)))

                tb.add_scalar("Test Loss", np.mean(total_loss), epoch)
                #tb.add_scalar("Testing Correct", total_correct, epoch)
                tb.add_scalar("Test Accuracy", total_accuracy, epoch)
                tb.add_scalar("Dummy Accuracy", np.floor(100.0*dummy_accuracy), epoch)
                tb.add_scalar("Mean test pixel dist", torch.mean(testPD), epoch)
                tb.close()

                if np.mean(total_loss) <= best_test_loss:
                    best_test_loss = np.mean(total_loss)
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': pipeline.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_test_loss': best_test_loss,
                                }, pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
                    print('Model saved')

            lastFolder = newFolder
