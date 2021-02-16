import os, cv2
from tqdm import tqdm
import torch, argparse
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pipeline_new import FusionPipeline, FINAL_DATASET
from helpers import Helpers
from variables import RootVariables

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()
    trim_frame_size = 150
    # model_checkpoint = 'pipeline_checkpoint.pth'
    # flownet_checkpoint = 'FlowNet2-S_checkpoint.pth.tar'
    # pipeline = FusionPipeline(args, flownet_checkpoint, trim_frame_size, device)
    # if Path(pipeline.var.root + model_checkpoint).is_file():
    #     checkpoint = torch.load(pipeline.var.root + model_checkpoint)
    #     pipeline.load_state_dict(checkpoint['model_state_dict'])
    #     print('Model loaded')

    utils = Helpers()
    var = RootVariables()
    criterion = nn.L1Loss()
    folder = 'test_InTheDeak_S2'
    frames, imu, targets = utils.load_datasets_folder(folder)
    # os.chdir(var.root)
    # imu_testing_feat = np.copy(imu)
    # imu_testing_feat = utils.standarization(imu_testing_feat)
    #
    # trainDataset = FINAL_DATASET(frames, imu_testing_feat, targets)
    # trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True)
    # tqdm_trainLoader = tqdm(trainLoader)
    #
    # num_samples, catList = 0, None
    # total_loss, total_correct, total_accuracy = 0.0, 0.0, 0.0
    # pipeline.eval()
    # with torch.no_grad():
    #     num_samples = 0
    #     total_loss, total_correct, total_accuracy = 0.0, 0.0, 0.0
    #     for batch_index, (frame_feat, imu_feat, labels) in enumerate(tqdm_trainLoader):
    #         num_samples += frame_feat.size(0)
    #         labels = labels[:,0,:]
    #         pred = pipeline(frame_feat, imu_feat).to(device)
    #         loss = criterion(pred*1000.0, (labels*1000.0).float())
    #         total_loss += loss.item()
    #         total_correct += pipeline.get_num_correct(pred, labels.float())
    #         total_accuracy = total_correct / num_samples
    #         tqdm_trainLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3}'.format(
    #             total_loss / num_samples, total_correct, 100.0*total_accuracy))
    #
    #         if batch_index == 0:
    #             catList = pred
    #         else:
    #             catList = torch.cat((catList, pred), axis=0)
    #
    #     torch.save(catList, pipeline.var.root + 'combined_' + folder[6:] + '_predictions.pt')

    print(frames.shape, imu.shape, targets.shape)
    print(targets[0], imu[0].shape, imu[0][0])
    os.chdir(var.root + folder)
    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(frame_count, fps, targets[0])
    capture.set(cv2.CAP_PROP_POS_FRAMES,trim_frame_size)
    ret, frame = capture.read()
    coordinate = torch.load(var.root + 'combined_' + folder[5:] + '_predictions.pt', map_location=torch.device('cpu'))
    coordinate = coordinate.detach().cpu().numpy()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('combined_output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
    # plt.scatter(0, 1080)
    # plt.scatter(1920, 0)
    for i in range(frame_count - 300):
        if ret == True:
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 512, 512)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # coordinate = sliced_gaze_dataset[i]
            try:
                gt_gaze_pts = targets[i][0]
                # gt_gaze_pts = np.sum(sliced_gaze_dataset[i], axis=0) / 4.0
                pred_gaze_pts = coordinate[i]
                padding_r = 100.0
                padding = 100.0
                # plt.scatter(int(pred_gaze_pts[0]*frame.shape[1]), int(pred_gaze_pts[1]*frame.shape[0]))

                start_point = (int(gt_gaze_pts[0]*frame.shape[1]) - int(padding), int(gt_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                end_point = (int(gt_gaze_pts[0]*frame.shape[1]) + int(padding), int(gt_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                pred_start_point = (int(pred_gaze_pts[0]*frame.shape[1]) - int(padding), int(pred_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                pred_end_point = (int(pred_gaze_pts[0]*frame.shape[1]) + int(padding), int(pred_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                #
                frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
                frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)

                frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1]),int(gt_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
                frame = cv2.circle(frame, (int(pred_gaze_pts[0]*frame.shape[1]),int(pred_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 255, 0), thickness=5)
            except:
                pass
            cv2.imshow('image', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            ret, frame = capture.read()

    # plt.show()
