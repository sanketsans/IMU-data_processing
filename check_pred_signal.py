import os, cv2
from tqdm import tqdm
import torch, argparse
from pathlib import Path
import numpy as np
from pipeline_new import FusionPipeline
from signal_pipeline import IMU_PIPELINE, IMU_DATASET
from prepare_dataset import IMU_GAZE_FRAME_DATASET, UNIFIED_DATASET

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trim_frame_size = 150
    model_checkpoint = 'signal_pipeline_checkpoint.pth'
    pipeline = IMU_PIPELINE(trim_frame_size, device)

    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded')

    uni_dataset = pipeline.prepare_dataset()
    uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    uni_gaze_dataset = uni_dataset.gaze_datasets
    print(len(uni_imu_dataset), len(uni_gaze_dataset), uni_gaze_dataset[0])


    start_index, end_index = 0, 0

    pipeline.eval()
    sliced_imu_dataset, sliced_gaze_dataset, sliced_frame_dataset = None, None, None
    catList = None
    for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
        if 'imu_' in subDir:
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_index = start_index + frame_count - trim_frame_size*2
            print(subDir, start_index, end_index)
            start_index = end_index

        if 'test_' in subDir:
            print(subDir)
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_index = start_index + frame_count - trim_frame_size*2
            sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()
            start_index = end_index

        # if 'val_' in subDir:
        #     print(subDir)
        #     with torch.no_grad():
        #         subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
        #         os.chdir(pipeline.var.root + subDir)
        #         capture = cv2.VideoCapture('scenevideo.mp4')
        #         frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #         end_index = start_index + frame_count - trim_frame_size*2
        #         sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
        #         sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()
        #         # print(sliced_gaze_dataset[0])
        #
        #         if not Path(pipeline.var.root + 'signal_' + subDir[4:-1] + '_predictions.pt').is_file():
        #             unified_dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
        #             unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
        #             tqdm_valLoader = tqdm(unified_dataloader)
        #             for batch_index, (imu_data, gaze_data) in enumerate(tqdm_valLoader):
        #                 gaze_data = torch.sum(gaze_data, axis=1) / 4.0
        #                 coordinates = pipeline(imu_data.float()).to(device)
        #
        #                 if batch_index == 0:
        #                     catList = coordinates
        #                 else:
        #                     catList = torch.cat((catList, coordinates), axis=0)
        #
        #             torch.save(catList, 'signal_' + subDir[:-1] + '_predictions.pt')
        #
        #     start_index = end_index

    # print(sliced_gaze_dataset[0], sliced_imu_dataset[0])
    print(sliced_gaze_dataset[0])
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
    # print(len(coordinate), len(sliced_gaze_dataset))

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('signal_output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
    # frame_count = 0
    # df_gaze = df_gaze.T
    for i in range(frame_count - 2):
        if ret == True:
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 512, 512)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # coordinate = sliced_gaze_dataset[i]
            gt_gaze_pts = np.sum(sliced_gaze_dataset[i], axis=0) / 4.0
            # pred_gaze_pts = coordinate[i]
            # for index, pt in enumerate(coordinate):
            #     try:
            #         (x, y) = pt[0], pt[1]
            #         frame = cv2.circle(frame, (int(x*frame.shape[1]),int(y*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
            #     except Exception as e:
            #         print(e)
            frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1]),int(gt_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
            # frame = cv2.circle(frame, (int(pred_gaze_pts[0]*frame.shape[1]),int(pred_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 255, 0), thickness=5)
            cv2.imshow('image', frame)
            # out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            ret, frame = capture.read()
