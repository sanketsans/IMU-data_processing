import cv2, os, sys
import numpy as np
sys.path.append('../')
from loader import JSON_LOADER
from variables import RootVariables

if __name__ == "__main__":
    var = RootVariables()
    folder = 'train_BookShelf_S1/'
    os.chdir(var.root + folder)
    capture = cv2.VideoCapture('scenevideo.mp4')
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    dataset = JSON_LOADER(folder)
    dataset.POP_GAZE_DATA(frame_count)
    gaze_arr = np.array(dataset.var.gaze_data).transpose()
    temp = np.zeros((frame_count*4-var.trim_frame_size*4*2, 2))
    temp[:,0] = gaze_arr[tuple([np.arange(var.trim_frame_size*4, frame_count*4 - var.trim_frame_size*4), [0]])]
    temp[:,1] = gaze_arr[tuple([np.arange(var.trim_frame_size*4, frame_count*4 - var.trim_frame_size*4), [1]])]
    print(temp[0], temp[4])

    index = 0
    capture.set(cv2.CAP_PROP_POS_FRAMES,var.trim_frame_size)
    ret, frame = capture.read()
    print(frame.shape)
    while ret:
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 512, 512)
        gaze = temp[4*index]
        try:
            frame = cv2.circle(frame, (int(gaze[0]*frame.shape[1]),int(gaze[1]*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
        except:
            pass
        print(gaze)
        index += 1
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(0)
        ret, frame = capture.read()
