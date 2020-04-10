from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tkinter import *
def knn(trainFeature , trainlable , testFeature , testLable,actualLable):

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(trainFeature,trainlable)
    predictLable= model.predict(testFeature)
    acc = accuracy_score(testLable, predictLable) * 100
    root = Tk()
    T = Text(root, height=6, width=30)
    root.title("Final OutPut")
    T.pack()
    T.insert(END, "Actual Class : ")
    T.insert(END, actualLable+1)
    T.insert(END, "\n\n")
    T.insert(END, "Predicted Class : ")
    T.insert(END, int(predictLable[0])+1)
    T.insert(END, "\n\n")
    T.insert(END, "Accuracy KNN : ")
    T.insert(END, acc)

    mainloop()