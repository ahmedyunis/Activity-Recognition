import os.path
import glob
import numpy as np
import HoG as hog
import SVM  as svm
import KNN as knn
def trainTestHog(ct , videoTest , lableTest):
    label = 0
    #train
    Train_Features=[]
    Train_Labels=[]
    finalTrain=[]
    finalLabel=[]

    file_type = "*.avi"
    parent_dir_train = "E:/second term/Computer Vision/project/final باذن الله/COM_VIS/DATA SET/TRAIN_CLASS"
    # CREATE TRAIN FEATURE

    for sub_dir in os.listdir(parent_dir_train):
        if os.path.isdir(os.path.join(parent_dir_train, sub_dir)):
            for vid in glob.glob(os.path.join(parent_dir_train, sub_dir, file_type)):
                Train_F, Train_L = hog.HOG(vid , label)
                Train_Features.append(Train_F)
                Train_Labels.append(Train_L)
        label = label + 1
    Train_Features = np.array(Train_Features)
    Train_Labels = np.array(Train_Labels)

    for i in range(len(Train_Features)):
        finalTrain+=Train_Features[i]
        finalLabel+=Train_Labels[i]
    finalTrain = np.array(finalTrain)
    finalLabel = np.array(finalLabel)

    # CREATE TEST FEATURE
    Test_Features , Test_Labels =hog.HOG(videoTest ,lableTest)
    Test_Features = np.array(Test_Features)
    Test_Labels = np.array(Test_Labels)
    if ct=='s' or ct=='S':
        svm.svm(finalTrain , finalLabel , Test_Features , Test_Labels , lableTest)
    else:
        knn.knn(finalTrain , finalLabel , Test_Features , Test_Labels , lableTest)