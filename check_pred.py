import os
from pipeline_new import FusionPipeline

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

    for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
        pipeline.init_stage()
        if 'imu_' in subDir:
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_index = start_index + frame_count - trim_frame_size*2 -1
        if 'test_' in subDir:
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_index = start_index + frame_count - trim_frame_size*2 -1

        start_index = end_index + 1
