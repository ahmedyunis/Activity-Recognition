import os.path
import glob
import numpy as np
import HoG as hog
import HoF as hof
import SVM  as svm
import KNN as knn
def trainTestHogHof(ct, videoTest , lableTest):
    label = 0
    # train
    Train_Features = []
    Train_Labels = []
    Train_HOF_Feature = []
    Train_HOF_Labels = []
    finalTrain = []
    finalLabel = []
    finalHofTrain = []
    finalHofLabel = []

    file_type = "*.avi"
    parent_dir_train = "E:/second term/Computer Vision/project/final باذن الله/COM_VIS/DATA SET/TRAIN_CLASS"

    # CREATE TRAIN FEATURE
    for sub_dir in os.listdir(parent_dir_train):
        if os.path.isdir(os.path.join(parent_dir_train, sub_dir)):
            for vid in glob.glob(os.path.join(parent_dir_train, sub_dir, file_type)):
                Train_F, Train_L = hog.HOG(vid,label)
                Train_HOF, Train_HOF_l = hof.HOF(vid,label)
                Train_Features.append(Train_F)
                Train_Labels.append(Train_L)
                Train_HOF_Feature.append(Train_HOF)
                Train_HOF_Labels.append(Train_HOF_l)
        label = label + 1

    Train_Features = np.array(Train_Features)
    Train_Labels = np.array(Train_Labels)
    Train_HOF_Feature = np.array(Train_HOF_Feature)
    Train_HOF_Labels = np.array(Train_HOF_Labels)

    for i in range(len(Train_Features)):
        finalTrain += Train_Features[i]
        finalLabel += Train_Labels[i]
        finalHofTrain += Train_HOF_Feature[i]
        finalHofLabel += Train_HOF_Labels[i]
    concatnationHogHof = finalTrain + finalHofTrain
    concatnationHogHofLable = finalLabel + finalHofLabel

    # CREATE TEST FEATURE
    Test_Features, Test_Labels = hog.HOG(videoTest,lableTest)
    Test_HOF_Feature, Test_HOF_Labels = hof.HOF(videoTest,lableTest)

    cHogHofTest = Test_Features + Test_HOF_Feature
    cHogHofLableTest = Test_Labels + Test_HOF_Labels
    concatnationHogHof = np.array(np.float32(np.nan_to_num(concatnationHogHof)))
    concatnationHogHofLable = np.array(concatnationHogHofLable)
    concatnationHogHofTest=np.array(np.float32(np.nan_to_num(cHogHofTest)))
    concatnationHogHofLableTest =np.array(cHogHofLableTest)
    if ct=='s' or ct=='S':
        svm.svm(concatnationHogHof , concatnationHogHofLable , concatnationHogHofTest ,concatnationHogHofLableTest , lableTest)
    else:
        knn.knn(concatnationHogHof , concatnationHogHofLable , concatnationHogHofTest ,concatnationHogHofLableTest,lableTest)