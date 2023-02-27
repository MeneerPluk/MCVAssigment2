import numpy as np
import bgSubstraction as bs
import cv2 as cv
import os
import random

lookuptable = dict()

for x in range(40):
    for y in range(40):
        for z in range(80):
            for c in range(1,5):
                objp = [x,y,z]

                path = f'data/cam{c}/config.xml'
                r = cv.FileStorage(path, cv.FileStorage_READ)
                tvecs = r.getNode('CameraTranslationVecs').mat()
                rvecs = r.getNode('CameraRotationVecs').mat()
                mtx = r.getNode('CameraIntrinsicMatrix').mat()
                dist = r.getNode('DistortionCoeffs').mat()
                Imgpt, jac = cv.projectPoints(np.float32([[objp[0]*25+12,objp[1]*25+12,objp[2]*25+12]]), rvecs, tvecs, mtx, dist)

                lookuptable[(objp[0],objp[1],objp[2],c)] = Imgpt


voxels = np.ones((40,40,80))
for c in range(1,5):
    path = os.path.abspath(f'data/cam{c}/video.avi')
    vid = cv.VideoCapture(path)
    nrOfFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)

    i = random.randint(0, nrOfFrames)
    vid.set(cv.CAP_PROP_POS_FRAMES, i)
    succes, img = vid.read()



    if succes:
        dilation = 2 if c == 2 else 0
        model = None
        if c == 1:
            model = bs.model1
        elif c==2:
            model = bs.model2
        elif c==3:
            model = bs.model3
        elif c==4:
            model = bs.model4

        img = bs.substractBackground(img, model, dilation)
        cv.imshow(f'{c}', img)
    for x in range(40):
        for y in range(40):
            for z in range(80):
                imgpt = lookuptable[(x,y,z,c)]
                if img[int(imgpt[0][0][1]),int(imgpt[0][0][0])] == 0:
                    voxels[x,y,z] = 0

voxelList = []
for x in range(40):
    for y in range(40):
        for z in range(80):
            if voxels[x,y,z] == 1:
                voxelList.append([x,z,-y])


