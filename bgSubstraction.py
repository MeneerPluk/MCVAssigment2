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


def testBackgroundModel(bgSubstractor, fgVidPath, dilation):
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
            contours, hierarchy  = cv.findContours(fgImg, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            list = [len(x) for x in contours]
            indexes = np.argsort(list,0)

            testimg = np.zeros_like(fgImg)

            fgImg = cv.drawContours(testimg, contours, indexes[-1] , (255,255,255), -1)

            kernel = np.ones((3,3),np.uint8) 
            fgImg = cv.dilate(fgImg,kernel, iterations = dilation)



            cv.imshow('bgImg', fgImg)
            cv.waitKey(1)





vidpath2 = os.path.abspath('data/cam4/video.avi')
vidpath = os.path.abspath('data/cam4/background.avi')
fgbg = cv.createBackgroundSubtractorMOG2(150, 100, True)
fgbg.setShadowValue(0)
fgbg.setShadowThreshold(0.4)


#cam 1 = 100, 0.5, no dilation
#cam 2 = 100, 0.42, with 2 iterations of dilation
#cam 3 = 100, 0.5, no dilation
#cam 4 = 100, 0.5, no dilation


trainBackgroundModel(fgbg,vidpath)
testBackgroundModel(fgbg, vidpath2, 0)