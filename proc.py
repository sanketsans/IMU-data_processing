import os
import json
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
from pathlib import Path
sys.path.append('../')
from helpers import Helpers
from variables import Variables

class BUILDING_DATASET:
    def __init__(self):
        self.utils = Helpers()
        self.var = Variables()
        try:
            g = os.system('gunzip gazedata.gz')
            i = os.system('gunzip imudata.gz')
        except Exception as e:
            pass
            # print(e)
        with open('gazedata') as f:
            for jsonObj in f:
                studentDict = json.loads(jsonObj)
                self.var.gaze_dataList.append(studentDict)

        with open('imudata') as f:
            for jsonObj in f:
                studentDict = json.loads(jsonObj)
                self.var.imu_dataList.append(studentDict)


    def POP_GAZE_DATA(self, return_val=False):
        ### GAZE DATA
        nT, oT = 1.9, 1.9
        for data in self.var.gaze_dataList:
            try:
                if(float(data['timestamp']) > 0.000000000 and float(data['timestamp']) < 600.0):
                    nT = self.utils.floor(data['timestamp'])
                    diff = round(nT - oT, 2)
                    self.var.gaze_data[0].append(data['data']['gaze2d'][0])
                    self.var.gaze_data[1].append(data['data']['gaze2d'][1])
                    self.var.timestamps_gaze.append(nT)
                    self.var.n_gaze_samples += 1
                    oT = nT
            except Exception as e:
                self.var.timestamps_gaze.append(nT)
                self.var.gaze_data[0].append(0.0)
                self.var.gaze_data[1].append(0.0)

        if return_val:
            return self.utils.get_sample_rate(self.var.timestamps_gaze)

    def POP_IMU_DATA(self, return_val=False):
        nT, oT = 0.0, 0.0

        for data in self.var.imu_dataList:
            try:
                if(float(data['timestamp']) > 0.000 and float(data['timestamp']) < 600.00):
                    nT = self.utils.floor(data['timestamp'])
                    diff = round((nT - oT), 2)
                    # print(nT)
                    # print(diff, round((nT-oT), 2))
                    self.var.imu_data_acc[0].append(data['data']['accelerometer'][0])
                    self.var.imu_data_acc[1].append(data['data']['accelerometer'][1])
                    self.var.imu_data_acc[2].append(data['data']['accelerometer'][2])

                    self.var.imu_data_gyro[0].append(data['data']['gyroscope'][0])
                    self.var.imu_data_gyro[1].append(data['data']['gyroscope'][1])
                    self.var.imu_data_gyro[2].append(data['data']['gyroscope'][2])

                    self.var.timestamps_imu.append(nT)
                    if (diff <= 0.01 and self.var.check_repeat==True):
                        self.utils.get_average_remove_dup(self.var.imu_data_acc[0], -2, -3)
                        self.utils.get_average_remove_dup(self.var.imu_data_acc[1], -2, -3)
                        self.utils.get_average_remove_dup(self.var.imu_data_acc[2], -2, -3)

                        self.utils.get_average_remove_dup(self.var.imu_data_gyro[0], -2, -3)
                        self.utils.get_average_remove_dup(self.var.imu_data_gyro[1], -2, -3)
                        self.utils.get_average_remove_dup(self.var.imu_data_gyro[2], -2, -3)

                        self.var.timestamps_imu.pop(len(self.var.timestamps_imu) - 3)
                        # print('Mid point resolved')

                        self.var.n_imu_samples -= 1
                        self.var.check_repeat = False
                        # try:
                        #     # if nT < 1.00 :
                        #     print('rmv dup', utils.floor(self.var.timestamps_gaze[self.var.gaze_data_index]), self.var.timestamps_imu[-1], diff)
                        #     self.var.gaze_data_index += 1
                        # except:
                        #     print(utils.floor(self.var.timestamps_gaze[self.var.gaze_data_index-1]),utils.floor(self.var.timestamps_imu[-1]))
                    elif (diff < 0.01):
                        self.var.check_repeat = True
                        # print('var.check_repeat is true now')
                    else:
                        pass
                        # try:
                        #     # if nT < 1.00 :
                        #     print('rmv dup', utils.floor(self.var.timestamps_gaze[self.var.gaze_data_index]), self.var.timestamps_imu[-1], diff)
                        #     self.var.gaze_data_index += 1
                        # except:
                        #     print(utils.floor(self.var.timestamps_gaze[self.var.gaze_data_index-1]),utils.floor(self.var.timestamps_imu[-1]))
                    self.var.n_imu_samples += 1
                    oT = nT
            except Exception as e:
                pass

        if return_val:
            return self.utils.get_sample_rate(self.var.timestamps_imu)
                #############################

if __name__ == "__main__":
    folder = sys.argv[1]
    dataset_folder = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'
    # os.chdir(dataset_folder)
    os.chdir(dataset_folder + folder + '/' if folder[-1]!='/' else (dataset_folder + folder))
    dataset = BUILDING_DATASET()
    if Path('gaze_file.csv').is_file():
        print('File exists')
    else:

        print(dataset.POP_GAZE_DATA())
        print(dataset.POP_IMU_DATA())
        print('IMU samples: {}, Gaze samples: {}'.format(dataset.var.n_imu_samples, dataset.var.n_gaze_samples))
    # fig = plt.figure()

# print(utils.get_sample_rate(var.timestamps_imu), len(var.timestamps_imu))
# print(utils.get_sample_rate(var.timestamps_gaze), len(var.timestamps_gaze))
# plt.subplot(221)
# # plt.stem(var.timestamps_imu, var.imu_data_acc[0], 'r')
# plt.plot(var.timestamps_imu, var.imu_data_acc[0], label='x-axis')
# # s = np.array(var.imu_data_acc[1])
# # plt.specgram(var.imu_data_acc[1], Fs = 1)
# # plt.title('matplotlib.pyplot.specgram() Example\n',
# #           fontsize = 14, fontweight ='bold')
# # plt.plot(var.timestamps_imu, var.imu_data_acc[1], label='y-axis')
# # plt.plot(var.timestamps_imu, var.imu_data_acc[2], label='z-axis')
# # plt.plot(var.timestamps_imu, roll)
# plt.legend()
#
# plt.subplot(222)
# plt.plot(var.timestamps_imu, var.imu_data_gyro[0], label='x-axis')
# # plt.plot(var.timestamps_imu, var.imu_data_gyro[1], label='y-axis')
# # plt.plot(var.timestamps_imu, var.imu_data_gyro[2], label='z-axis')
# # plt.plot(var.timestamps_imu, pitch)
# plt.legend()
#
# # print('Total samples: {}, y[0]: {}, y[1]: {}'.format(len(x), len(y[0]), len(y[1])))
# plt.subplot(223)
# plt.plot(var.timestamps_gaze, var.gaze_data[0])
#
# plt.subplot(224)
# plt.plot(var.timestamps_gaze, var.gaze_data[1])
#
# plt.show()
