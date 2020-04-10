import os.path
import glob
import numpy as np
import HoF as hof
import SVM  as svm
import KNN as knn
def trainTestHof(ct, videoTest , lableTest):
    label = 0
    # train
    Train_HOF_Feature = []
    Train_HOF_Labels = []
    finalHofTrain = []
    finalHofLabel = []

    file_type = "*.avi"
    parent_dir_train = "E:/second term/Computer Vision/project/final باذن الله/COM_VIS/DATA SET/TRAIN_CLASS"

    # CREATE TRAIN FEATURE
    for sub_dir in os.listdir(parent_dir_train):
        if os.path.isdir(os.path.join(parent_dir_train, sub_dir)):
            for vid in glob.glob(os.path.join(parent_dir_train, sub_dir, file_type)):
                Train_HOF, Train_HOF_l = hof.HOF(vid , label)
                Train_HOF_Feature.append(Train_HOF)
                Train_HOF_Labels.append(Train_HOF_l)
        label = label + 1
        print("Class (", label , ") was finished")
    Train_HOF_Feature = np.array(Train_HOF_Feature)
    Train_HOF_Labels = np.array(Train_HOF_Labels)

    for i in range(len(Train_HOF_Feature)):
        finalHofTrain += Train_HOF_Feature[i]
        finalHofLabel += Train_HOF_Labels[i]

    finalHofTrain = np.array(np.float32(np.nan_to_num(finalHofTrain)))
    finalHofLabel = np.array(finalHofLabel)

    # CREATE TEST FEATURE
    Test_HOF_Feature, Test_HOF_Labels = hof.HOF(videoTest , lableTest)
    Test_HOF_Feature = np.array(np.float32(np.nan_to_num(Test_HOF_Feature)))
    Test_HOF_Labels = np.array(Test_HOF_Labels)
    if ct=="s" or ct=="S":
        svm.svm(finalHofTrain , finalHofLabel , Test_HOF_Feature , Test_HOF_Labels ,lableTest)
    else:
        knn.knn(finalHofTrain , finalHofLabel , Test_HOF_Feature , Test_HOF_Labels , lableTest)
