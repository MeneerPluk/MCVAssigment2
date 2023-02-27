import cv2 as cv
import os
import numpy as np
import random

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


def testBackgroundModel(bgSubstractor, fgVidPath, dilation):
    vid = cv.VideoCapture(fgVidPath)
    nrOfFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)

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
            cv.waitKey(10)

def substractBackground(img, bgSubstractor, dilation):
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

    return fgImg



vidpath1 = os.path.abspath('data/cam1/background.avi')
model1 = cv.createBackgroundSubtractorMOG2(150, 100, True)
model1.setShadowValue(0)
model1.setShadowThreshold(0.5)

vidpath2 = os.path.abspath('data/cam2/background.avi')
model2 = cv.createBackgroundSubtractorMOG2(150, 100, True)
model2.setShadowValue(0)
model2.setShadowThreshold(0.42)

vidpath3 = os.path.abspath('data/cam3/background.avi')
model3 = cv.createBackgroundSubtractorMOG2(150, 100, True)
model3.setShadowValue(0)
model3.setShadowThreshold(0.5)

vidpath4 = os.path.abspath('data/cam4/background.avi')
model4 = cv.createBackgroundSubtractorMOG2(150, 100, True)
model4.setShadowValue(0)
model4.setShadowThreshold(0.5)

modelList = [model1, model2, model3, model4]
vidpathList = [vidpath1,vidpath2,vidpath3,vidpath4]

print("please wait while the 4 background models get trained!")
for  mod,vidpath in zip(modelList,vidpathList):
    trainBackgroundModel(mod,vidpath)
    print("training model done")



#cam 1 = 100, 0.5, no dilation
#cam 2 = 100, 0.42, with 2 iterations of dilation
#cam 3 = 100, 0.5, no dilation
#cam 4 = 100, 0.5, no dilation


#g = substractBackground(img, model2, 2)


