import cv2
import glob
import numpy as np
import math as math

HoF_FEATURE_NO = []

def cell_histogram(cell_direction, cell_magnitude):
    hist_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
    cell_hist = np.zeros(shape=(hist_bins.size))
    cell_size = cell_direction.shape[0]
    for r in range(cell_size - 1):
        for c in range(cell_size - 1):
            current_direct = cell_direction[r, c]
            current_magnit = cell_magnitude[r, c]
            diff = np.abs(current_direct - hist_bins)
            first_idx = np.where(diff == np.min(diff))[0][0]
            if first_idx == 8:  # hist_bins.size - 1
                temp = hist_bins[[(first_idx - 1), (0)]]
                temp2 = np.abs(current_direct - temp)
                res = np.where(temp2 == np.min(temp2))[0][0]
            else:
                temp = hist_bins[[(first_idx - 1), (first_idx + 1)]]
                temp2 = np.abs(current_direct - temp)
                res = np.where(temp2 == np.min(temp2))[0][0]
            if res == 0 and first_idx != 0:
                second_idx = first_idx - 1

            else:
                second_idx = first_idx + 1

            first_value = hist_bins[second_idx]
            second_value = hist_bins[first_idx]
            x = 0
            y=0
            x = cell_hist[first_idx] + (
                    np.abs(current_direct - first_value) / (180.0 / hist_bins.size)) * current_magnit
            cell_hist[first_idx]=x
            y = cell_hist[second_idx] + (
                    np.abs(current_direct - second_value) / (180.0 / hist_bins.size)) * current_magnit
            cell_hist[second_idx]=y
    return (cell_hist)


def HOF(vid , lable):
    Labels_HOF = []
    hofFeature = []
    finalHofFeature=[]
    sec = 0
    frameRate = 0.5
    cap = cv2.VideoCapture(vid)
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (64, 128), interpolation=cv2.INTER_AREA)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    label_Hof = []
    angle=np.zeros([8,8])
    magnitude=np.zeros([8,8])
    while (1):
        Big_Hof=[]
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame2 = cap.read()
        if ret:
            HOF_NORM_FEATURE = []
            sec = sec + frameRate
            sec = round(sec, 2)
            frame2 = cv2.resize(frame2, (64, 128), interpolation=cv2.INTER_AREA)
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            for i in range(0, frame1.shape[0], 8):
                HOF_Feature = []

                for j in range(0, frame1.shape[1], 8):
                    prvs1 = prvs[i:i + 8, j:j + 8]
                    next1 = next[i:i + 8, j:j + 8]
                    flow = cv2.calcOpticalFlowFarneback(prvs1, next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    angle = ang * 180 / np.pi / 2
                    magnitude=cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    feature = cell_histogram(angle , magnitude)
                    HOF_Feature.append(feature)
                Big_Hof.append(HOF_Feature)
            global HoF_FEATURE_NO
            Big_Hof = np.array(Big_Hof)
            for i in range(len(Big_Hof)-1):
                HoF_FEATURE_NO=[]
                for j in range(len(Big_Hof[0])-1):
                    x=[Big_Hof[i][j]] + [Big_Hof[i][j+1] ]+[Big_Hof[i+1][j] ]+[Big_Hof[i+1][j+1] ]
                    x=np.array(x)
                    x=x.reshape(36,1)
                    multArr= x*x
                    sumMultArr= sum(multArr)
                    res= math.sqrt(sumMultArr)
                    divideRes= x/res
                    HoF_FEATURE_NO.append(divideRes)
                HOF_NORM_FEATURE.append(HoF_FEATURE_NO)
            Labels_HOF.append(lable)
            prvs = next
            hofFeature = np.array(HOF_NORM_FEATURE)
            hofFeature = np.reshape(hofFeature, (1, np.product(hofFeature.shape)))[0]
            finalHofFeature.append(hofFeature)
        else:
            break
    return finalHofFeature, Labels_HOF
