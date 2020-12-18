import sys
import torch
import argparse
from imu_encoder import IMU_ENCODER
from vis_encoder import VIS_ENCODER
from gaze_plotter import GET_DATAFRAME_FILES
from getDataset import IMUDataset, ImageDataset
from variables import RootVariables

if __name__ == "__main__":
    folder = 'BookShelf_S1/'
    var = RootVariables()

    dataset = IMUDataset(var.root, folder)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=var.batch_size)
    a = iter(trainLoader)
    data = next(a)
    # model = IMU_ENCODER(var.input_size, var.hidden_size, var.num_layers, var.num_classes).to(device)
    # scores = model(data.float())
    # print(model, scores.shape)

    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)

    args = parser.parse_args()
    dataset = ImageDataset(var.root, folder)
    dataset.populate_data(dataset.first_frame)
    # torch.save(dataset.stack_frames, var.root + folder + 'stack_frames.pt')
    print(len(dataset.stack_frames))
    # trainLoader = torch.utils.data.DataLoader(dataset, batch_size=var.batch_size)
    # a = iter(trainLoader)
    # imgs = next(a)
    ## load model without batch norm
    # checkpoint_path = var.root + "FlowNet2-S_checkpoint.pth.tar"
    #
    # img_enc = VIS_ENCODER(args, checkpoint_path)
    # output = img_enc.run_model(imgs)
    # print(imgs.shape, dataset[5].shape)
