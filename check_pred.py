import os
from pipeline_new import FusionPipeline
from prepare_dataset import IMU_GAZE_FRAME_DATASET, UNIFIED_DATASET

if __name__ == "__main__":
    folder = 'imu_BookShelf_S1/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    model_checkpoint = 'pipeline_checkpoint.pth'
    flownet_checkpoint = 'FlowNet2-S_checkpoint.pth.tar'
    trim_frame_size = 150
    # current_loss_mean_train, current_loss_mean_val, current_loss_mean_test = 0.0, 0.0,  0.0
    pipeline = FusionPipeline(args, flownet_checkpoint, trim_frame_size, device)

    if Path(pipeline.var.root + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_loss = checkpoint['loss']
        print('Model loaded')

    uni_dataset = pipeline.prepare_dataset()
    uni_imu_dataset = uni_dataset.imu_datasets      ## will already be standarized
    uni_gaze_dataset = uni_dataset.gaze_datasets

    start_index, end_index = 0, 0

    pipeline.eval()

    for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
        pipeline.init_stage()
        if 'imu_' in subDir:
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_index = start_index + frame_count - trim_frame_size*2 -1
        if 'test_' in subDir:
            with torch.no_grad():
                subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                os.chdir(pipeline.var.root + subDir)
                capture = cv2.VideoCapture('scenevideo.mp4')
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                end_index = start_index + frame_count - trim_frame_size*2 -1
                sliced_imu_dataset = uni_imu_dataset[start_index: end_index].detach().cpu().numpy()
                sliced_gaze_dataset = uni_gaze_dataset[start_index: end_index].detach().cpu().numpy()

                # sliced_frame_dataset = torch.load('framesExtracted_data_' + str(trim_frame_size) + '.pt')
                sliced_frame_dataset = np.load('framesExtracted_data_' + str(trim_frame_size) + '.npy', mmap_mode='r')
                unified_dataset = UNIFIED_DATASET(sliced_frame_dataset, sliced_imu_dataset, sliced_gaze_dataset, device)
                unified_dataloader = torch.utils.data.DataLoader(unified_dataset, batch_size=pipeline.var.batch_size, num_workers=0, drop_last=True)
                tqdm_valLoader = tqdm(unified_dataloader)
                for batch_index, (frame_data, imu_data, gaze_data) in enumerate(tqdm_valLoader):
                    gaze_data = torch.sum(gaze_data, axis=1) / 8.0
                    coordinates = pipeline(frame_data, imu_data).to(device)

        start_index = end_index + 1

    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(frame_count, fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES,trim_frame_size)
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(frame.shape)

    # df_gaze = df_gaze.T
    for i in range(frame_count):
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 512, 512)
            cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = capture.read()
