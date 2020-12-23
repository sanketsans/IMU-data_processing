from variables import Variables
import os, ast
import numpy as np
import cv2, torch, itertools
from gaze_plotter import GET_DATAFRAME_FILES

def get_imu_data_pts_from_dataframe_values(dataframe_values):
    data_pts_x, data_pts_y, data_pts_z = [], [], []
    g_x, g_y, g_z = [], [], []
    for data in dataframe_values:
        (acc, gyro) = ast.literal_eval(data)
        data_pt = np.array(acc+gyro)
        data_pt[1] += 9.80665
        data_pts_x.append(np.round(data_pt[0], 3))
        data_pts_y.append(np.round(data_pt[1], 3))
        data_pts_z.append(np.round(data_pt[2], 3))

        g_x.append(np.round(data_pt[3], 3))
        g_y.append(np.round(data_pt[3], 3))
        g_z.append(np.round(data_pt[3], 3))

    return data_pts_x, data_pts_y, data_pts_z, g_x, g_y, g_z

if __name__ == "__main__":
    folder = 'imu_InTheDeak_S2/'
    video_file = 'scenevideo.mp4'
    var = Variables()
    os.chdir(var.root + folder)
    capture = cv2.VideoCapture(video_file)
    total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    dataframes = GET_DATAFRAME_FILES(folder, total_frame_count)
    imu_data_frame, dataset = dataframes.get_imu_dataframe()
    dataset.POP_IMU_DATA()
    acc_x = dataset.var.imu_data_acc[0]
    s = (acc_x[0:4]-np.min(acc_x))/(np.max(acc_x)-np.min(acc_x))
    print(acc_x[0:4], s)

    ## IMU
    # imu_data_frame = dataframes.get_imu_dataframe().T
    # print(list(itertools.chain(*imu_data_frame.values[1].tolist()))
