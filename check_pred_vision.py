import os, cv2
from tqdm import tqdm
import torch, argparse
from pathlib import Path
import numpy as np
from pipeline_new import FusionPipeline
from vision_pipeline import VISION_PIPELINE, VISION_DATASET
from prepare_dataset import IMU_GAZE_FRAME_DATASET, UNIFIED_DATASET

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    model_checkpoint = 'hidden_256_relu_50e_VP.pth'
    flownet_checkpoint = 'FlowNet2-S_checkpoint.pth.tar'

    trim_frame_size = 150
    pipeline = VISION_PIPELINE(args, flownet_checkpoint, device)

    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded')

    uni_dataset = pipeline.prepare_dataset()
    uni_gaze_dataset = uni_dataset.gaze_datasets

    start_index, end_index = 0, 0

    pipeline.eval()
    with torch.no_grad():
        sliced_imu_dataset, sliced_gaze_dataset, sliced_frame_dataset = None, None, None
        catList = None
        for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
            if 'imu_' in subDir:
                print(subDir)
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                end_index = start_index + frame_count - trim_frame_size*2
                if 'imu_smallGroupMeeting_S4csdvsdb' in subDir:
                    sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index]
                    print(sliced_gaze_dataset[0])

                    if not Path(pipeline.var.root + 'vision_' + subDir[4:-1] + '_predictions.pt').is_file():
                        sliced_frame_dataset = np.load(str(pipeline.var.frame_size) + '_framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')

                        unified_dataset = VISION_DATASET(sliced_frame_dataset, sliced_gaze_dataset, device)
                        unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                        tqdm_valLoader = tqdm(unified_dataloader)
                        for batch_index, (frame_data, gaze_data) in enumerate(tqdm_valLoader):
                            gaze_data = torch.sum(gaze_data, axis=1) / 4.0
                            coordinates = pipeline(frame_data).to(device)

                            if batch_index == 0:
                                catList = coordinates
                            else:
                                catList = torch.cat((catList, coordinates), axis=0)

                        torch.save(catList, pipeline.var.root + 'vision_' + subDir[4:-1] + '_predictions.pt')

                    break

                start_index = end_index

            if 'test_' in subDir:
                print(subDir)
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                end_index = start_index + frame_count - trim_frame_size*2
                if 'test_InTheDeak_S2' in subDir:
                    sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index]
                    print(sliced_gaze_dataset[0])

                    if not Path(pipeline.var.root + 'vision_' + subDir[4:-1] + '_predictions.pt').is_file():
                        sliced_frame_dataset = np.load(str(pipeline.var.frame_size) + '_framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')

                        unified_dataset = VISION_DATASET(sliced_frame_dataset, sliced_gaze_dataset, device)
                        unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                        tqdm_valLoader = tqdm(unified_dataloader)
                        for batch_index, (frame_data, gaze_data) in enumerate(tqdm_valLoader):
                            gaze_data = torch.sum(gaze_data, axis=1) / 4.0
                            coordinates = pipeline(frame_data).to(device)

                            if batch_index == 0:
                                catList = coordinates
                            else:
                                catList = torch.cat((catList, coordinates), axis=0)

                        torch.save(catList, pipeline.var.root + 'vision_' + subDir[4:-1] + '_predictions.pt')

                    break

                start_index = end_index

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
    for i in range(frame_count - 1):
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
