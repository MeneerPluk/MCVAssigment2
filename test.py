import bgSubstraction as bS
import os
import cv2 as cv
import random

path = os.path.abspath('data/cam2/video.avi')
vid = cv.VideoCapture(path)
nrOfFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)

i = random.randint(0, nrOfFrames)
vid.set(cv.CAP_PROP_POS_FRAMES, i)
succes, img = vid.read()



if succes:
    g = bS.substractBackground(img, bS.model2, 2)
    cv.imshow('g', g)
    cv.waitKey(-1)


