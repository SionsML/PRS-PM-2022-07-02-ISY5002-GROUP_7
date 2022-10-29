from tkinter import *
import tkinter.messagebox
import pickle



#internal library
from collectFaceImage import collectFaceImageAndStore
import constant
from facialLandmark import facialLandmarkRecognition
from cnnImageAlertnessDetection import alertnessDetection
from lrcnAlertnessDetection import main
from convLSTMAlertnessDetection import main
from rpConvLSTM_camera import rpmain
#from collectFaceImage import keepAndTrain




root = Tk()
root.geometry('500x320')
frame = Frame(root, relief = RIDGE, borderwidth = 2)
frame.pack(fill = BOTH, expand = 1)
root.title('Driver Alertness Detection System')
frame.config(background='#87CEEB')	
label = Label(frame, text="Driver Alertness Detection System",bg='#87CEEB',font=('Constantia 22 bold'))
label.pack(side = TOP)
filename = PhotoImage(file="background.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)
known_face_encodings = []

def loadFaceRecognitionModel():
   try:
      with open(constant.model_face_recognition, 'rb') as f:
         known_face_encodings = pickle.load(f)
   except:
      known_face_encodings = []


def creatorInfo():
   tkinter.messagebox.showinfo("Contributors","Hwang Sion\n" "Prerak Agarwal\n" "Santi\n" "Zhang Junfeng")

def aboutWindow():
   tkinter.messagebox.showinfo("About",'Driver Alertness Detection System, with multiple AI/Machine Learning Models')

def supportWindow():
   tkinter.messagebox.showinfo("Support","[Click on various options available.]\n" "[First option allows frontal monitor.]\n" "[Second option allows side monitor.]\n" )
                                                                  

menu = Menu(root)
root.config(menu = menu)

sub_menu1 = Menu(menu)
menu.add_cascade(label="Help",menu=sub_menu1)
sub_menu1.add_command(label="Support",command=supportWindow)

sub_menu2 = Menu(menu)
menu.add_cascade(label="About",menu=sub_menu2)
sub_menu2.add_command(label="Info",command=aboutWindow)
sub_menu2.add_command(label="Contributors",command=creatorInfo)

loadFaceRecognitionModel()

def CNNModel():
    #eye_model = keras.models.load_model('CNN_Model_Utility/best_model.h5')
    facialLandmarkRecognition(known_face_encodings)    

     

def alertnessDetectionModel():
   # Second model Code to be here
   alertnessDetection(known_face_encodings)
   print("test second model")
   
def rnnAlertnessDetectionModel():
   # third model Code to be here
   if constant.device_type == 'rp':
      rpmain(known_face_encodings)
   else:
      main(known_face_encodings)
   print("test third model")


def collectFaceData():
   # This is to collect face of driver

    print("Collect My Face")   
    collectFaceImageAndStore()



# ===========================================================================================================
# Button Configurations

but1=Button(frame, padx = 6, pady = 6, width = 46, bg = 'white',fg = 'black',relief = GROOVE,command = CNNModel, text = 'Front Camera Sleepiness Monitoring',font = ('Constantia 13 bold'))
but1.place(x = 5,y = 80)

but2=Button(frame, padx = 6,pady = 6, width = 46, bg = 'white', fg = 'black', relief = GROOVE, command = alertnessDetectionModel, text = 'Side Camera Alertness Monitoring', font = ('Constantia 13 bold'))
but2.place(x = 5,y = 140)

but3=Button(frame, padx = 6, pady = 6, width=46, bg = 'white', fg = 'black', relief = GROOVE, command = rnnAlertnessDetectionModel, text = 'Front Camera Alertness Monitoring', font = ('Constantia 13 bold'))
but3.place(x = 5,y = 200)

but4=Button(frame, padx = 6, pady = 6, width=46, bg = 'white', fg = 'black', relief = GROOVE, command = collectFaceData, text = 'Register Me!', font = ('Constantia 13 bold'))
but4.place(x = 5,y = 260)

root.mainloop()

# ===========================================================================================================
