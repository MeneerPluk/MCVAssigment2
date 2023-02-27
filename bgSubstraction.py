import cv2 as cv
import os
import numpy as np

def trainBackgroundModel(bgSubstractor, bgVidPath):
    vid = cv.VideoCapture(bgVidPath)
    nrOfFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)

    for i in range(0,int(nrOfFrames)):
        vid.set(cv.CAP_PROP_POS_FRAMES, i)
        succes, img = vid.read()
        if succes:
            #preprossesing with a gaussian blur:
            img = cv.GaussianBlur(img,(3,3), 0)
            #training the background model:
            bgSubstractor.apply(img, None, -1)
            bgImg = bgSubstractor.getBackgroundImage()
            cv.imshow('bgImg', bgImg)
            cv.waitKey(10)


def testBackgroundModel(bgSubstractor, fgVidPath):
    vid = cv.VideoCapture(fgVidPath)
    nrOfFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)
    kernel = np.zeros((5,5))
    for i in range(0,int(nrOfFrames)):
        vid.set(cv.CAP_PROP_POS_FRAMES, i)
        succes, img = vid.read()
        if succes:
            #preprossessing with a gaussian blur:
            blur = cv.GaussianBlur(img,(3,3), 0)
            #substracting the background:
            fgImg = bgSubstractor.apply(blur, None, 0)
            #finding the contour:
            contours, hierarchy  = cv.findContours(fgImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            testimg = np.zeros_like(fgImg)

            fgImg = cv.drawContours(testimg, contours, (len(contours)-1) , (255,0,0), -1)

            cv.imshow('bgImg', fgImg)
            cv.waitKey(250)





vidpath2 = os.path.abspath('data/cam1/video.avi')
vidpath = os.path.abspath('data/cam1/background.avi')
fgbg = cv.createBackgroundSubtractorMOG2(150, 100, True)
fgbg.setShadowValue(0)




trainBackgroundModel(fgbg,vidpath)
testBackgroundModel(fgbg, vidpath2)