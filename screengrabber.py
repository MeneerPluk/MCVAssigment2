import sys
import os
import glob
import cv2 as cv
import random

screens = 25
def captureIntrinsics(camnum):
    camfolder = 'data/cam' + str(camnum) + '/'
    vidpath = os.path.abspath(camfolder + 'intrinsics.avi')
    exportfolder = os.path.abspath(camfolder + 'intrinsics')
    if not os.path.exists(exportfolder):
        os.mkdir(exportfolder)
    else:
        ims = glob.glob(exportfolder + '/*')
        for im in ims:
            os.remove(im)

    vid = cv.VideoCapture(vidpath)
    frames = vid.get(cv.CAP_PROP_FRAME_COUNT)
    grabwindow = int(frames / screens)

    for i in range(0, screens):
        randomFrame = random.randint(i * grabwindow, (i+1) * grabwindow)
        vid.set(cv.CAP_PROP_POS_FRAMES, randomFrame)

        success, image = vid.read()
        if success:
            cv.imshow("Camera " + str(camnum), image)
            cv.waitKey(100)
            cv.imwrite(exportfolder + '/' + str(i) + '.jpg', image)

for i in range(1, 5):
    captureIntrinsics(i)