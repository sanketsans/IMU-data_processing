import os
import json
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
sys.path.append('../')
from gaze_data import helpers, variables

utils = helpers.Helpers()
var = variables.Variables()
nT, oT = 1.9, 1.9
roll, pitch = [], [] ## angle at which head moves vertically(roll)
folder = sys.argv[1]

dataset_folder = '/home/sans/Downloads/gaze_data/'
# os.chdir(dataset_folder)
os.chdir(dataset_folder + folder + '/' if folder[-1]!='/' else (dataset_folder + folder))
fig = plt.figure()

try:
    os.system('gunzip ' + 'gazedata.gz')
    os.system('gunzip ' + 'imudata.gz')
except:
    pass

with open('gazedata') as f:
    for jsonObj in f:
        studentDict = json.loads(jsonObj)
        var.gaze_dataList.append(studentDict)

# oldList = []
for data in var.gaze_dataList:
    try:
        if(float(data['timestamp']) > 6.000000000 and float(data['timestamp']) < 600.0):
            nT = utils.floor(data['timestamp'])
            diff = round(nT - oT, 2)

            var.gaze_data[0].append(data['data']['gaze2d'][0])
            var.gaze_data[1].append(data['data']['gaze2d'][1])
            var.timestamps_gaze.append(nT)
            var.n_gaze_samples += 1
            oT = nT
            # oldList = data
    except Exception as e:
        # pass
        # print(data['timestamp'], e)
        var.timestamps_gaze.append(nT)
        var.gaze_data[0].append(0.0)
        var.gaze_data[1].append(0.0)

with open('imudata') as f:
    for jsonObj in f:
        studentDict = json.loads(jsonObj)
        var.imu_dataList.append(studentDict)

oT, nT = 0.0, 0.0
for data in var.imu_dataList:
    try:
        if(float(data['timestamp']) > 5.000 and float(data['timestamp']) < 600.00):
            nT = utils.floor(data['timestamp'])
            diff = round((nT - oT), 2)
            # print(diff, round((nT-oT), 2))
            var.imu_data_acc[0].append(data['data']['accelerometer'][0])
            var.imu_data_acc[1].append(data['data']['accelerometer'][1])
            var.imu_data_acc[2].append(data['data']['accelerometer'][2])

            var.imu_data_gyro[0].append(data['data']['gyroscope'][0])
            var.imu_data_gyro[1].append(data['data']['gyroscope'][1])
            var.imu_data_gyro[2].append(data['data']['gyroscope'][2])

            var.timestamps_imu.append(nT)
            if (diff <= 0.01 and var.check_repeat==True):
                utils.get_average_remove_dup(var.imu_data_acc[0], -2, -3)
                utils.get_average_remove_dup(var.imu_data_acc[1], -2, -3)
                utils.get_average_remove_dup(var.imu_data_acc[2], -2, -3)

                utils.get_average_remove_dup(var.imu_data_gyro[0], -2, -3)
                utils.get_average_remove_dup(var.imu_data_gyro[1], -2, -3)
                utils.get_average_remove_dup(var.imu_data_gyro[2], -2, -3)

                var.timestamps_imu.pop(len(var.timestamps_imu) - 3)
                # print('Mid point resolved')

                var.n_imu_samples -= 1
                var.check_repeat = False
                try:
                    roll.append(math.atan2(var.imu_data_acc[1][-1], var.imu_data_acc[2][-1])* 180/math.pi)

                    pitch.append(math.atan2(-var.imu_data_acc[0][-1], math.sqrt(var.imu_data_acc[1][-1]*var.imu_data_acc[1][-1] + var.imu_data_acc[2][-1]*var.imu_data_acc[2][-1]))*180/math.pi)
                    # print(utils.floor(var.timestamps_gaze[var.gaze_data_index]), var.timestamps_imu[-1], diff)
                    var.gaze_data_index += 1
                except:
                    print(utils.floor(var.timestamps_gaze[var.gaze_data_index-1]),utils.floor(var.timestamps_imu[-1]))
            elif (diff < 0.01):
                var.check_repeat = True
                # print('var.check_repeat is true now')
            else:
                try:
                    roll.append(math.atan2(var.imu_data_acc[1][-1], var.imu_data_acc[2][-1])*180/math.pi)

                    pitch.append(math.atan2(-var.imu_data_acc[0][-1], math.sqrt(var.imu_data_acc[1][-1]*var.imu_data_acc[1][-1] + var.imu_data_acc[2][-1]*var.imu_data_acc[2][-1]))*180/math.pi)
                    # print(utils.floor(var.timestamps_gaze[var.gaze_data_index]), var.timestamps_imu[-1], diff)
                    var.gaze_data_index += 1
                except:
                    print(utils.floor(var.timestamps_gaze[var.gaze_data_index-1]),utils.floor(var.timestamps_imu[-1]))
            var.n_imu_samples += 1
            oT = nT
    except Exception as e:
        pass
        # print(e, 'Hello', data)

print('IMU samples: {}, Gaze samples: {}'.format(var.n_imu_samples, var.n_gaze_samples))

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
