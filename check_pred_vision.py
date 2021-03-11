import os, cv2
from tqdm import tqdm
import torch, argparse
from pathlib import Path
import numpy as np
from helpers import Helpers
import torch.nn as nn
from vision_pipeline import VISION_PIPELINE, FINAL_DATASET

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_folder = 'test_BookShelf_S1'
    model_checkpoint = 'vision_checkpoint_' + test_folder[5:] + '.pth'
    flownet_checkpoint = 'flownets_EPE1.951.pth.tar'
    # flownet_checkpoint = 'FlowNet2-SD_checkpoint.pth.tar'

    arg = 'del'
    n_epochs = 1
    toggle = 0
    trim_frame_size = 150
    pipeline = VISION_PIPELINE(flownet_checkpoint, device)
    print(pipeline)
    criterion = nn.L1Loss()

    if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    utils = Helpers(test_folder)
    _, _, _, testing_target = utils.load_datasets()
    os.chdir(pipeline.var.root)

    pipeline.eval()
    with torch.no_grad():
        testDataset = FINAL_DATASET('test_', testing_target)
        testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)

        tqdm_testLoader = tqdm(testLoader)
        num_samples = 0
        total_loss, total_correct, total_accuracy = [], 0.0, 0.0
        finalPred, testPD = [], []
        for batch_index, (feat, labels) in enumerate(tqdm_testLoader):
            num_samples += feat.size(0)
            labels = labels[:,0,:]

            pred = pipeline(feat.float()).to(device)
            loss = criterion(pred, labels.float())

            pred, labels = pipeline.get_original_coordinates(pred, labels)

            dist = torch.cdist(pred, labels.float(), p=2)
            if batch_index > 0:
                testPD = torch.cat((testPD, dist), 0)
                finalPred = torch.cat((finalPred, pred), 0)
            else:
                testPD = dist
                finalPred = pred

            total_loss.append(loss.detach().item())
            total_correct += pipeline.get_num_correct(pred, labels.float())
            total_accuracy = total_correct / num_samples
            tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3} {}'.format(
                np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(testPD)))

        finalPred = finalPred.cpu().detach().numpy()

        with open(self.var.root + 'final_pred_Book.npy') as f:
            np.save(f, finalPred)
            f.close()

    print(sliced_gaze_dataset[0])
    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(frame_count, fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES,trim_frame_size)
    ret, frame = capture.read()
    subDir = 'test_InTheDeak_S2/'
    coordinate = torch.load(pipeline.var.root + 'vision_' + subDir[4:-1] + '_predictions.pt', map_location=torch.device('cpu'))
    coordinate = coordinate.detach().cpu().numpy()
    print(len(coordinate), len(sliced_gaze_dataset))
    #
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('vision_output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
    # frame_count = 0
    # df_gaze = df_gaze.T
    for i in range(0):
        if ret == True:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 512, 512)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # coordinate = sliced_gaze_dataset[i]
            try:
                gt_gaze_pts = np.sum(sliced_gaze_dataset[i], axis=0) / 4.0
                pred_gaze_pts = coordinate[i] / 1000.0
                start_point = (int(gt_gaze_pts[0]*frame.shape[1]) - int(0.02*frame.shape[1]), int(gt_gaze_pts[1]*frame.shape[0]) + int(0.02*frame.shape[0]))
                end_point = (int(gt_gaze_pts[0]*frame.shape[1]) + int(0.02*frame.shape[1]), int(gt_gaze_pts[1]*frame.shape[0]) - int(0.02*frame.shape[0]))
                pred_start_point = (int(pred_gaze_pts[0]*frame.shape[1]) - int(0.02*frame.shape[1]), int(pred_gaze_pts[1]*frame.shape[0]) + int(0.02*frame.shape[0]))
                pred_end_point = (int(pred_gaze_pts[0]*frame.shape[1]) + int(0.02*frame.shape[1]), int(pred_gaze_pts[1]*frame.shape[0]) - int(0.02*frame.shape[0]))

                frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
                frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)
                frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1]) ,int(gt_gaze_pts[1]*frame.shape[0])), radius=5, color=(255, 0, 0), thickness=5)
                frame = cv2.circle(frame, (int(pred_gaze_pts[0]*frame.shape[1]),int(pred_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 0), thickness=5)
            except :
                pass
            cv2.imshow('image', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            ret, frame = capture.read()
