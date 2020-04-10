from tkinter import *
from tkinter import filedialog
import HoGTrainTest as GTT
import HoFTrainTest as FTT
import HoGcomHoF as HCH
top = Tk()
top.geometry("300x220" )
top.title("WELCOM"  )
top.configure(bg ='gray')

var = StringVar()
l = Label (top, textvariable=var,bg='gray', font=('times', 14))
var.set ("Type Classifier")
l.pack()
l.place(x=30,y=160)

e = Entry(top)
e.pack()
e.focus_set()
e.place(x=160,y=160)

#///////////////////////////////////////////
videoName =0
def readVideo():
    top.videoname = filedialog.askopenfilename(initialdir="E:/second term/Computer Vision/project/final باذن الله/COM_VIS/DATA SET/CLASSES",title="Select folder",
                                              filetypes=(("all folder", "*.*"), ("avi folder", "*.avi")))
    global videoPath
    global lable
    videoPath = top.videoname
    videoName = videoPath.split('_')[-3]
    if videoName =='Archery':
        lable =0
    elif videoName=='Basketball':
        lable=1
    elif videoName=='Bowling':
        lable=2
    elif videoName=='BoxingPunchingBag':
        lable=3
    elif videoName=='Diving':
        lable=4
    elif videoName=='HorseRiding':
        lable=5
    elif videoName=='UnevenBars':
        lable=6
    else:
        print("Bad Input Video")
def hog():
    ct = e.get()
    GTT.trainTestHog(ct[0] ,videoPath ,lable )
def hof():
    ct = e.get()
    FTT.trainTestHof(ct[0],videoPath ,lable)
def combhoghof():
    ct=e.get()
    HCH.trainTestHogHof(ct[0],videoPath ,lable)
B= Button(top, text="Select Test Video" ,command =readVideo,font=('helvetica', 12 , 'bold'))
B.place(x=80,y=40)

B= Button(top, text="HOG" ,command =hog,font=('helvetica', 12 , 'bold'))
B.place(x=30,y=100)

B= Button(top, text="HOF" ,command =hof,font=('helvetica', 12 , 'bold'))
B.place(x=120,y=100)

B= Button(top, text="HOG + HOF" ,command =combhoghof,font=('helvetica', 12 , 'bold'))
B.place(x=190,y=100)

top.mainloop()