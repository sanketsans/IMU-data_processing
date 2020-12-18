import os
root = '/home/sans/Downloads/gaze_data/'

for files in os.listdir(root):
    if 'imu_' in files:
        print(files)
