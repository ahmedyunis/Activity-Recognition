import numpy as np
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
from tkinter import *

def svm(trainFeature , trainlable , testFeature , testLable , actualLable):
    SVM = cv2.ml.SVM_create()
    SVM.setType(cv2.ml.SVM_C_SVC)  # AS Classifier
    SVM.setKernel(cv2.ml.SVM_RBF)
    SVM.setGamma(0.01)
    SVM.train(trainFeature, cv2.ml.ROW_SAMPLE, trainlable)
    Labels_pred = SVM.predict(testFeature)[1].ravel()
    Labels_pred = np.array(Labels_pred)
    acc=accuracy_score(testLable, Labels_pred) * 100
    root = Tk()
    T = Text(root, height=6, width=30)
    T.pack()
    root.title("Final OutPut")
    T.insert(END ,"Actual Class : ")
    T.insert(END , actualLable+1)
    T.insert(END , "\n\n")
    T.insert(END ,"Predicted Class : ")
    T.insert(END , int(Labels_pred[0])+1)
    T.insert(END , "\n\n")
    T.insert(END , "Accuracy SVM : ")
    T.insert(END , acc)
    mainloop()