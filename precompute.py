import sys
import os
import glob
import cv2 as cv
import random

screens = 25
def captureIntrinsics(camnum):
    """
    """
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
        term = False
        count = 0
        while not term and count < 15:
            count+=1
            randomFrame = random.randint(i * grabwindow, (i+1) * grabwindow)
            vid.set(cv.CAP_PROP_POS_FRAMES, randomFrame)

            success, image = vid.read()
            if success:
                ret, _ = cv.findChessboardCorners(image, (8,6), cv.CALIB_CB_FAST_CHECK)
                term = ret
        if term:
            cv.imshow("Camera " + str(camnum), image)
            cv.waitKey(100)
            cv.imwrite(exportfolder + '/' + str(i) + '.jpg', image)

def GetAverageFrame(camnum):
    """
    """
    camfolder = 'data/cam' + str(camnum) + '/'
    vidpath = os.path.abspath(camfolder + 'background.avi')
    vid = cv.VideoCapture(vidpath)
    avgFrame = None
    frames = vid.get(cv.CAP_PROP_FRAME_COUNT)

    while vid.isOpened():
        _, frame = vid.read()
        if frame is None:
            break
        if avgFrame is None:
            avgFrame = frame.astype(float)
        else:
            avgFrame += frame.astype(float)

    avgFrame /= frames
    avgFrame = avgFrame.astype('uint8')
    print(f'Saving average of cam {camnum}')
    cv.imwrite(camfolder + 'background.jpg', avgFrame)

def captureExtrinsics(camnum):
    """
    """
    camfolder = 'data/cam' + str(camnum) + '/'
    vidpath = os.path.abspath(camfolder + 'checkerboard.avi')
    exportfolder = os.path.abspath(camfolder + 'extrinsic')
    if not os.path.exists(exportfolder):
        os.mkdir(exportfolder)
    else:
        ims = glob.glob(exportfolder + '/*')
        for im in ims:
            os.remove(im)

    vid = cv.VideoCapture(vidpath)
    frames = vid.get(cv.CAP_PROP_FRAME_COUNT)
    randomFrame = random.randint(0, frames)
    vid.set(cv.CAP_PROP_POS_FRAMES, randomFrame)

    success, image = vid.read()
    if success:
        cv.imshow("Camera " + str(camnum), image)
        cv.waitKey(100)
        cv.imwrite(exportfolder + '/' + "0" + '.jpg', image)
    

if __name__ == "__main__":
    for i in range(1, 5):
        GetAverageFrame(i)
        #captureExtrinsics(i)