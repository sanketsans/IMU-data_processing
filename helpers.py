import math, os, cv2
from variables import RootVariables
from torch.utils.data import Dataset
import numpy as np
from prepare_dataset import IMU_GAZE_FRAME_DATASET

class ALIGN_DATASET(Dataset):
    def __init__(self, imu_data, gaze_data):
        self.frame_data = frame_dataset
        self.gaze_data = gaze_dataset
        self.per_file_frame = []
        self.per_file_gaze = []
        checkedLast = False
        for i in range(len(self.gaze_data) - 2):
            index = i + 1
            while True:
                check = np.isnan(self.gaze_data[index])
                if check.any():
                    index = (index - 1) if checkedLast else (index + 1)
                    if index == self.__len__():
                        checkedLast = True
                else:
                    break
            imu_index = 75 + index
            catIMUData = self.imu_data[imu_index-25]
            for i in range(25):
                catIMUData = np.concatenate((catIMUData, self.imu_data[imu_index-1+i]), axis=0)
            # for i in range(1, 3):
            #     catIMUData = np.concatenate((catIMUData, self.imu_data[imu_index+i]), axis=0)

            self.per_file_frame.append(self.frame_data[index-1])
            self.per_file_imu.append(catIMUData)
            self.per_file_gaze.append(self.gaze_data[index])

        self.per_file_frame = np.array(self.per_file_frame)
        self.per_file_imu = np.array(self.per_file_imu)
        self.per_file_gaze = np.array(self.per_file_gaze)

    def __len__(self):
        return len(self.gaze_data) - 1

class Helpers:
    def __init__(self):
        self.value = 0
        self.var = RootVariables()

        self.dataset = IMU_GAZE_FRAME_DATASET(self.var.root, self.var.frame_size, self.var.trim_frame_size)
        self.train_imu_dataset, self.test_imu_dataset = self.dataset.imu_train_datasets, self.dataset.imu_test_datasets
        self.train_gaze_dataset, self.test_gaze_dataset = self.dataset.gaze_train_datasets, self.dataset.gaze_test_datasets

        self.train_folders_num, self.test_folders_num = 0, 0
        self.gaze_start_index, self.gaze_end_index = 0, 0
        self.imu_start_index, self.imu_end_index = 0, 0
        self.toggle = 0

    def floor(self, value):
        return math.floor(value*100)/100.0


    def get_sample_rate(self, samples):
        total_sample = 0.0
        not_consistent = 0
        curr_bin = math.floor(samples[0])
        count = 0
        sample_rate = {}
        not_cons_sample_rate = {}
        for sample in samples:
            total_sample += sample - total_sample
            if total_sample > float(curr_bin)+0.99:
                sample_rate[curr_bin] = count
                # if (count != 100):
                #     not_consistent += 1
                #     not_cons_sample_rate[curr_bin] = count
                curr_bin = math.floor(total_sample)
                count = 0
            count += 1

        sample_rate[curr_bin] = count
        # if (count != 100):
        #     not_consistent += 1
        #     not_cons_sample_rate[curr_bin] = count

        return sample_rate ##if you want all the sample rates.
        # return not_cons_sample_rate, not_consistent

    def get_average_remove_dup(self, samples, avg_ind, rm_ind):
        samples[avg_ind] = (samples[avg_ind] + samples[rm_ind]) / 2.0
        samples.pop(len(samples) - abs(rm_ind))

    def load_datasets(self):
        frame_training_feat, frame_testing_feat = None, None
        imu_training_feat, imu_testing_feat = None, None
        training_target, testing_target = None, None

        check = True if Path(self.var.root + str(self.var.frame_size) + '_training_framesExtracted_data_feat_' + str(self.var.trim_frame_size) + '.npy').is_file() else False
        if check :
            frame_training_feat = np.load(self.var.root + str(self.var.frame_size) + '_frames_training_feat_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
            frame_testing_feat = np.load(self.var.root + str(self.var.frame_size) + '_frame_testing_feat_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
            imu_training_feat = np.load(self.var.root + str(self.var.frame_size) + '_imu_training_feat_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
            imu_testing_feat = np.load(self.var.root + str(self.var.frame_size) + '_imu_testing_feat_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
            training_target = np.load(self.var.root + str(self.var.frame_size) + '_gaze_training_target_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
            testing_target = np.load(self.var.root + str(self.var.frame_size) + '_gaze_testing_target_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')

        else:
            for index, subDir in enumerate(sorted(os.listdir(self.var.root))):
                if 'train_' in subDir:
                    if toggle != 1:
                        toggle = 1
                        self.gaze_start_index, self.imu_start_index = 0, 0
                    self.train_folders_num += 1
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(self.var.root + subDir)
                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.gaze_end_index = self.gaze_start_index + frame_count - self.var.trim_frame_size*2
                    self.imu_end_index = self.imu_start_index + frame_count - self.var.trim_frame_size
                    sliced_frame_dataset = np.load(str(self.var.frame_size) + '_framesExtracted_data_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
                    sliced_imu_dataset = self.train_imu_dataset[self.imu_start_index: self.imu_end_index]
                    sliced_gaze_dataset = self.train_gaze_dataset[self.gaze_start_index: self.gaze_end_index]
                    data = ALIGN_DATASET(sliced_imu_dataset, sliced_gaze_dataset)

                    if self.train_folders_num > 1:
                        frame_training_feat, imu_training_feat, training_target = np.concatenate((frame_training_feat, data.per_file_frame), axis=0),
                                                                                    np.concatenate((imu_training_feat, data.per_file_imu), axis=0),
                                                                                        np.concatenate((training_target, data.per_file_gaze), axis=0)
                    else:
                        frame_training_feat, imu_training_feat, training_target = data.per_file_frame, data.per_file_imu, data.per_file_gaze

                    self.gaze_start_index = self.gaze_end_index
                    self.imu_start_index = self.imu_end_index

                if 'test_' in subDir:
                    if toggle != -1:
                        toggle = -1
                        self.gaze_start_index, self.imu_start_index = 0, 0

                    self.test_folders_num += 1
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(self.var.root + subDir)
                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.gaze_end_index = self.gaze_start_index + frame_count - self.var.trim_frame_size*2
                    self.imu_end_index = self.imu_start_index + frame_count - self.var.trim_frame_size
                    sliced_frame_dataset = np.load(str(self.var.frame_size) + '_framesExtracted_data_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
                    sliced_imu_dataset = self.test_imu_dataset[self.imu_start_index: self.imu_end_index]
                    sliced_gaze_dataset = self.test_gaze_dataset[self.gaze_start_index: self.gaze_end_index]
                    data = ALIGN_DATASET(sliced_imu_dataset, sliced_gaze_dataset)

                    if self.test_folders_num > 1:
                        frame_testing_feat, imu_testing_feat, testing_target = np.concatenate((frame_testing_feat, data.per_file_frame), axis=0),
                                                                                 np.concatenate((imu_testing_feat, data.per_file_imu), axis=0),
                                                                                    np.concatenate((testing_target, data.per_file_gaze), axis=0)
                    else:
                        frame_testing_feat, imu_testing_feat, testing_target = data.per_file_frame,
                        data.per_file_imu, data.per_file_gaze

                    self.gaze_start_index = self.gaze_end_index
                    self.imu_start_index = self.imu_end_index

            with open(self.var.root + str(self.var.frame_size) + '_frames_training_feat_' + str(self.var.trim_frame_size) + '.npy', 'wb') as f:
                np.save(f, frame_training_feat)
                f.close()
            with open(self.var.root + str(self.var.frame_size) + '_frame_testing_feat_' + str(self.var.trim_frame_size) + '.npy', 'wb') as f:
                np.save(f, frame_testing_feat)
                f.close()
            with open(self.var.root + str(self.var.frame_size) + '_imu_training_feat_' + str(self.var.trim_frame_size) + '.npy', 'wb') as f:
                np.save(f, imu_training_feat)
                f.close()
            with open(self.var.root + str(self.var.frame_size) + '_imu_testing_feat_' + str(self.var.trim_frame_size) + '.npy', 'wb') as f:
                np.save(f, imu_testing_feat)
                f.close()
            with open(self.var.root + str(self.var.frame_size) + '_gaze_training_target_' + str(self.var.trim_frame_size) + '.npy', 'wb') as f:
                np.save(f, training_target)
                f.close()
            with open(self.var.root + str(self.var.frame_size) + '_gaze_testing_target_' + str(self.var.trim_frame_size) + '.npy', 'wb') as f:
                np.save(f, testing_target)
                f.close()

        return frame_training_feat, frame_testing_feat, imu_training_feat, imu_testing_feat, training_target, testing_target


if __name__ == "__main__":
    utils = Helpers()
