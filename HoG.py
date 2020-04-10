import os
import os.path
import cv2
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import math as math

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hogParams = {'winStride':(8, 8) , 'padding': (16, 16), 'scale': 1.05}


def HOG(vid , label):
    Features = []
    Labels = []
    vidcap = cv2.VideoCapture(vid)
    count = 0
    sec = 0
    while True:
        frameRate = 0.25  # //it will capture image in each 0.5 second
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = vidcap.read(0)
        if ret:
            cv2.imwrite("frame%d.jpg" % count, frame)  # save frame as JPEG file
        count += 1
        if not ret:
            break
        sec = sec + frameRate
        sec = round(sec, 2)
        (rects, weights) = hog.detectMultiScale(frame, **hogParams)
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        frame = cv2.resize(frame, (64, 128), interpolation=cv2.INTER_AREA)
        descriptor = hog.compute(frame)
        if descriptor is None:
            descriptor = []
        else:
            descriptor = descriptor.ravel()  # flattened array convert 2D to 1D

        Features.append(descriptor)
        Labels.append(label)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    return (Features, Labels)
