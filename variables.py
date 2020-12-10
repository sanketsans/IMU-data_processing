class Variables:
    def __init__(self):
        self.gaze_dataList = []     ## list to handle data with multiple single element json items.
        self.timestamps_gaze = []   ## timestamp of data
        self.gaze_data = [[], []]   ## 2D co-ordinates of gaze2D
        self.n_gaze_samples = 0     ## number of samples for corresponding sec.
        self.gaze_data_index = 0    ## used to print the timestamp

        self.imu_dataList = []
        self.timestamps_imu = []
        self.imu_data_acc = [[], [], []]
        self.imu_data_gyro = [[], [], []]
        self.n_imu_samples = 0
        self.check_repeat = False   ## to check if repeation.

if __name__ =='__main__':
    var = Variables()
