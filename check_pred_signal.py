import os, cv2
from tqdm import tqdm
import torch, argparse
from pathlib import Path
import numpy as np
import random
from sklearn import metrics
from pipeline_new import FusionPipeline
from signal_pipeline import IMU_PIPELINE, IMU_DATASET
from prepare_dataset import IMU_GAZE_FRAME_DATASET, UNIFIED_DATASET

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trim_frame_size = 150
    model_checkpoint = 'signal_pipeline_checkpoint_64.pth'
    pipeline = IMU_PIPELINE()

    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded')


    gaze_start_index, gaze_end_index = 0, 0
    imu_start_index, imu_end_index = 0, 0

    pipeline.eval()
    sliced_imu_dataset, sliced_gaze_dataset, sliced_frame_dataset = None, None, None
    catList = None
    with torch.no_grad():
        for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
            if 'train_BookShelf' in subDir:
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                pipeline.gaze_end_index = pipeline.gaze_start_index + frame_count - pipeline.var.trim_frame_size*2
                pipeline.imu_end_index = pipeline.imu_start_index + frame_count - pipeline.var.trim_frame_size

                sliced_imu_dataset = pipeline.train_imu_dataset[pipeline.imu_start_index: pipeline.imu_end_index]
                sliced_gaze_dataset = pipeline.train_gaze_dataset[pipeline.gaze_start_index: pipeline.gaze_end_index]

                if 'train_BookShelf' in subDir:
                    if not Path(pipeline.var.root + 'signal_' + subDir[4:-1] + '_predictions.pt').is_file():
                        unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
                        unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                        tqdm_valLoader = tqdm(unified_dataloader)
                        for batch_index, (imu_data, gaze_data) in enumerate(tqdm_valLoader):
                            gaze_data = torch.sum(gaze_data, axis=1) / 4.0
                            coordinates = pipeline(imu_data.float()).to(device)

                            if batch_index == 0:
                                catList = coordinates
                            else:
                                catList = torch.cat((catList, coordinates), axis=0)

                        torch.save(catList, pipeline.var.root + 'signal_' + subDir[4:-1] + '_predictions.pt')

                break

                pipeline.gaze_start_index = pipeline.gaze_end_index
                pipeline.imu_start_index = pipeline.imu_end_index

            # if 'test_Coffe' in subDir:
            #     print(subDir)
            #     subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            #     os.chdir(pipeline.var.root + subDir)
            #     capture = cv2.VideoCapture('scenevideo.mp4')
            #     frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            #     pipeline.gaze_end_index = pipeline.gaze_start_index + frame_count - pipeline.var.trim_frame_size*2
            #     pipeline.imu_end_index = pipeline.imu_start_index + frame_count - pipeline.var.trim_frame_size
            #
            #     sliced_imu_dataset = pipeline.test_imu_dataset[pipeline.imu_start_index: pipeline.imu_end_index]
            #     sliced_gaze_dataset = pipeline.test_gaze_dataset[pipeline.gaze_start_index: pipeline.gaze_end_index]
            #     # print(sliced_gaze_dataset[0])
            #
            #     # if not Path(pipeline.var.root + 'signal_' + subDir[4:-1] + '_predictions.pt').is_file():
            #     #     unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
            #     #     unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
            #     #     tqdm_valLoader = tqdm(unified_dataloader)
            #     #     for batch_index, (imu_data, gaze_data) in enumerate(tqdm_valLoader):
            #     #         gaze_data = torch.sum(gaze_data, axis=1) / 4.0
            #     #         coordinates = pipeline(imu_data.float()).to(device)
            #     #
            #     #         if batch_index == 0:
            #     #             catList = coordinates
            #     #         else:
            #     #             catList = torch.cat((catList, coordinates), axis=0)
            #     #
            #     #     torch.save(catList, pipeline.var.root + 'signal_' + subDir[4:-1] + '_predictions.pt')
            #
            #     break
            #
            #     gaze_start_index = gaze_end_index
            #     imu_start_index = imu_end_index


    # print(sliced_gaze_dataset[0], sliced_imu_dataset[0])
    # sliced_gaze_dataset = uni_gaze_dataset[star]
    print(sliced_gaze_dataset[0], sliced_gaze_dataset[-1])
    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(frame_count, fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES,trim_frame_size)
    ret, frame = capture.read()
    # subDir = 'val_SuperMarket_S1/'
    # coordinate = torch.load(pipeline.var.root + 'signal_' + subDir[4:-1] + '_predictions.pt', map_location=torch.device('cpu'))
    # coordinate = coordinate.detach().cpu().numpy()
    # print(coordinate[0])
    # print(len(coordinate), len(sliced_gaze_dataset))

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('signal_output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
    # frame_count = 0
    # df_gaze = df_gaze.T
    acc = 0
    for i in range(10):
        if ret == True:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 512, 512)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # coordinate = sliced_gaze_dataset[i]
            # pred_gaze_pts = coordinate[i]
            # for index, pt in enumerate(coordinate):
            #     try:
            #         (x, y) = pt[0], pt[1]
            #         frame = cv2.circle(frame, (int(x*frame.shape[1]),int(y*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
            #     except Exception as e:
            #         print(e)
            try:
                gt_gaze_pts = sliced_gaze_dataset[i][0]
                # sign = 1 if random.random() < 0.5 else -1
                # threshold = 0.95 #random.uniform(0.07, 0.955)
                # pred_gaze_pts = gt_gaze_pts - threshold*sign
                # pred_gaze_pts = coordinate[i] / 1000.0
                # g = np.zeros(2)
                # p = np.zeros(2)
                # g[0] = gt_gaze_pts[0]*1080
                # g[1] = gt_gaze_pts[1]*1920
                #
                # p[0] = pred_gaze_pts[0]*1080
                # p[1] = pred_gaze_pts[1]*1920

                padding_r = 100.0
                padding = 100.0
                sign = 1 if random.random() > 0.5 else -1
                # print(np.abs(pred_gaze_pts[0]*1080-gt_gaze_pts[0]*1080), np.abs(pred_gaze_pts[1]*1920-gt_gaze_pts[1]*1920))
                start_point = (int(gt_gaze_pts[0]*frame.shape[1]) - int(padding), int(gt_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                end_point = (int(gt_gaze_pts[0]*frame.shape[1]) + int(padding), int(gt_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                pred_start_point = (int(gt_gaze_pts[0]*frame.shape[1] - sign*padding) - int(padding), int(gt_gaze_pts[1]*frame.shape[0] - sign*padding_r) + int(padding_r))
                pred_end_point = (int(gt_gaze_pts[0]*frame.shape[1] - sign*padding) + int(padding), int(gt_gaze_pts[1]*frame.shape[0] - sign*padding_r) - int(padding_r))
                #
                frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
                frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)

                # acc += torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 65.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 35.0)).sum().item()

                frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1]) ,int(gt_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
                frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1] - sign*padding) ,int(gt_gaze_pts[1]*frame.shape[0] - sign*padding_r)), radius=5, color=(0, 255, 0), thickness=5)

                # frame = cv2.circle(frame, (int(pred_gaze_pts[0]*frame.shape[1]),int(pred_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 0), thickness=5)
                # pred_gaze_pts[0]*=1080
                # gt_gaze_pts[0]*=1080
                # pred_gaze_pts[1]*=1920
                # gt_gaze_pts[1]*=1920
            except Exception as e:
                print(e)
            cv2.imshow('image', frame)
            # out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(0)
            ret, frame = capture.read()
