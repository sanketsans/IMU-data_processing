from variables import RootVariables
if __name__ == "__main__":
    var = RootVariables()
    subDir = 'imu_BookShelf_S1/'
    os.chdir(var.root + subDir)
    # os.chdir(dataset_folder)

    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(frame_count, fps)
    ret, frame = capture.read()
    print(frame.shape)

    # df_gaze = df_gaze.T
    for i in range(frame_count):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 512, 512)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = capture.read()

    # count = 0
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
    # for i in range(length):
    #     if ret == True:
    #         # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #         # cv2.resizeWindow('image', 600,600)
    #         # image = cv2.circle(frame, (int(x*frame.shape[0]),int(y*frame.shape[1])), radius=5, color=(0, 0, 255), thickness=5)
    #
    #         coordinate = df_gaze.iloc[:,count]
    #         for index, pt in enumerate(coordinate):
    #             try:
    #                 (x, y) = ast.literal_eval(pt)
    #                 frame = cv2.circle(frame, (int(x*frame.shape[1]),int(y*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
    #             except Exception as e:
    #                 print(e)
    #             # pt = pt.strip('()')     ## 1315 frame, no gaze point ## 1298
    #             # (x, y) = tuple(map(float, pt.split(', ')))
    #         print(coordinate)
    #         out.write(frame)
    #         cv2.imshow('image', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #         # cv2.waitKey(0)
    #         ret, frame = capture.read()
    #         count += 1
    #     else :
    #         break
