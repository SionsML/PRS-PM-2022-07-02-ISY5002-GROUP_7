from tkinter import *
import tkinter.messagebox

import cv2
import numpy as np
from playsound import playsound
from PIL import Image, ImageDraw
import face_recognition
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import os
# from tensorflow import keras
from tflite_runtime.interpreter import Interpreter

import numpy as np
from playsound import playsound
import pygame


root = Tk()
root.geometry('500x300')
frame = Frame(root, relief = RIDGE, borderwidth = 2)
frame.pack(fill = BOTH, expand = 1)
root.title('Driver Alertness Detection System')
frame.config(background='#87CEEB')	
label = Label(frame, text="Driver Alertness Detection System",bg='#87CEEB',font=('Constantia 22 bold'))
label.pack(side = TOP)
filename = PhotoImage(file="background.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)

#set music
pygame.mixer.init()


# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_MODEL_FPS = 5  # Ensure the input images are fed to the model at this fps.
_MODEL_FPS_ERROR_RANGE = 0.1  # Acceptable error range in fps.


# ===========================================================================================================
# CNN_Model Methods Start

#eye_model = keras.models.load_model('best_model_test.h5')

def classify(img):
    # Load the TFLite model and allocate tensors.
    interpreter = Interpreter(model_path="CNN_Model_Utility/best_model_test.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test the model on random input data.
    #img = cv2.resize(img,(80,80),3)


    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = np.array(img, dtype=np.float32)
    input_tensor= np.array(input_data)
    #input_tensor= np.array(np.expand_dims(input_data,0))
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    top_k = np.argmax(results)
    return results

"""
    Utility method for eye cropper that calculates eye cordinates are referenced/adapted from here:
    Stewart, D. (2021). Preventing Drowsy-Driving Accidents Using Convolutional Neural Networks. Towards Data Science. 
    https://towardsdatascience.com/drowsiness-detection-using-convolutional-neural-networks-face-recognition-and-tensorflow-56cdfc8315ad 
"""

def crop_eye(frame):

    # Variable to save cordinates of face features
    face_feature_list = face_recognition.face_landmarks(frame)

    # List for eye coordinates 
    try:
        eye = face_feature_list[0]['left_eye']
    except:
        try:
            eye = face_feature_list[0]['right_eye']
        except:
            return
    
    # Get maximum and minimum coordinates of eye
    maxium_x = max([coordinate[0] for coordinate in eye])
    minimum_x = min([coordinate[0] for coordinate in eye])
    maxium_y = max([coordinate[1] for coordinate in eye])
    minimum_y = min([coordinate[1] for coordinate in eye])

    # Set x and y range
    range_x = maxium_x - minimum_x
    range_y = maxium_y - minimum_y

    # To ensure full eye is captured, add 50% buffer to axis with the larger range 
    # and match with smaller range
    if range_x > range_y:
        right = round(0.5 * range_x) + maxium_x
        left = minimum_x - round(0.5 * range_x)
        bottom = round((((right-left) - range_y))/2) + maxium_y
        top = minimum_y - round((((right-left) - range_y))/2)
    else:
        bottom = round(0.5 * range_y) + maxium_y
        top = minimum_y - round(0.5 * range_y)
        right = round((((bottom-top) - range_x))/2) + maxium_x
        left = minimum_x - round((((bottom-top) - range_x))/2)
    
    # Crop the images based on above caculations
    crop = frame[top:(bottom + 1), left:(right + 1)]

    # Do image resizing
    crop = cv2.resize(crop, (224,224))
    output = crop.reshape(-1, 224, 224, 3)

    return output

"""
    Utility methods for eye aspect ratio, lip distance are referenced/adapted from here:
    SPARKLERS : We Are The Makers. (2019, December 24). Realtime Drowsiness and Yawn Detection using Python in Raspberry Pi or any other PC [Video]. YouTube. 
    https://www.youtube.com/watch?v=RDuLqCT5RxY&t=533s 
""" 

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    left_Eye = shape[left_Start:left_End]
    right_Eye = shape[right_Start:right_End]

    left_EAR = eye_aspect_ratio(left_Eye)
    right_EAR = eye_aspect_ratio(right_Eye)

    ear = (left_EAR + right_EAR) / 2.0
    return (ear, left_Eye, right_Eye) 

def lip_distance(shape):
    upper_lip = shape[50:53]
    upper_lip = np.concatenate((upper_lip, shape[61:64]))

    lower_lip = shape[56:59]
    lower_lip = np.concatenate((lower_lip, shape[65:68]))

    upper_mean = np.mean(upper_lip, axis=0)
    lower_mean = np.mean(lower_lip, axis=0)

    distance = abs(upper_mean[1] - lower_mean[1])
    return distance

# ===========================================================================================================
# CNN_Model Methods End
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


def CNNModel():
    #eye_model = keras.models.load_model('CNN_Model_Utility/best_model.h5')
    
    detector = cv2.CascadeClassifier("CNN_Model_Utility/haarcascade_frontalface_default.xml")    
    predictor = dlib.shape_predictor('CNN_Model_Utility/shape_predictor_68_face_landmarks.dat')

    # Start webcam
    cap = cv2.VideoCapture('/home/pi/Documents/videostream_distraction/yawning.avi') # Can try -1 if 0 doesn't work
    counter, fps, last_inference_start_time, time_per_infer = 0, 0, 0, 0

    if not cap.isOpened():
        raise IOError('Cannot open webcam')

    # Set counter
    counter = 0
    yawnCounter = 0

    frame_rate = 10
    previous = 0

    # Set Threshold for Yawning
    yawn_Threshold = 25

    # create a while loop that runs while webcam is in use
    while cap.isOpened():
        res, image = cap.read()
        #ret, frame = cap.read()
        #frame = imutils.resize(frame, width = 600)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #time_passed = time.time() - previous
        current_frame_start_time = time.time()
        diff = current_frame_start_time - last_inference_start_time
        
        
        image = cv2.flip(image,1)
        print('Start====')
        print(diff)
        print(_MODEL_FPS)
        print(_MODEL_FPS_ERROR_RANGE)
        

        if diff * _MODEL_FPS >= (1 - _MODEL_FPS_ERROR_RANGE):
            #previous = time.time()
             
            last_inference_start_time = current_frame_start_time
            fps = 1.0 / diff
             
            frame = imutils.resize(image, width = 600)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             
            time_per_infer = time.time() - current_frame_start_time
            #rects = detector(gray, 0)
            rects = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            #for rect in rects:
            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
                
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                eye = final_ear(shape)
                ear = eye[0]
                leftEye = eye [1]
                rightEye = eye[2]

                distance = lip_distance(shape)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (255, 255, 255), 1)

                eye_image_to_predict = crop_eye(frame)
                try:
                    eye_image_to_predict = eye_image_to_predict/255.0
                except:
                    continue

                # Get prediction from model
                # prediction = eye_model.predict(eye_image_to_predict)
                prediction = classify(eye_image_to_predict)

                w1,h1 = 200,100
                # From model prediction, display either "Eyes Open" or "Eyes Closed"
                if prediction < 0.5:
                    counter = 0
                    cv2.putText(frame, 'Eyes Open', (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255,0),2)
                    
                else:
                    counter = counter + 1
                    cv2.putText(frame, 'Eyes Closed', (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0 , (0, 0,255),2)

                    if counter > 5:

                        cv2.putText(frame, 'DRIVER SLEEPING', (20, 130), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 2)
                        k = cv2.waitKey(1)
                        #playsound("CNN_Model_Utility/doorbell.wav", False)
                        #playsound("CNN_Model_Utility/wake_up.mp3", False)
                        pygame.mixer.music.load("CNN_Model_Utility/doorbell.wav")
                        pygame.mixer.music.play(2)                        
                        counter = 1
                        continue
                
                if (distance > yawn_Threshold):
                    yawnCounter = yawnCounter + 1
                    cv2.putText(frame, "Yawn Alert", (230, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    
                    if yawnCounter > 10:
                        #playsound("CNN_Model_Utility/yawn.mp3", False)
                        pygame.mixer.music.load("CNN_Model_Utility/yawn.wav")
                        pygame.mixer.music.play(2)                            
                        yawnCounter = 0

                cv2.putText(frame, "Yawn Ratio: {:.1f}".format(distance), (310, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255,255), 2)
        
        # Notes: Frames that aren't fed to the model are still displayed to make the
        # video look smooth. We'll show classification results from the latest
        # classification run on the screen.
        # Show the FPS .
        fps_text = 'Current FPS = {0:.1f}. Expect: {1}'.format(fps, _MODEL_FPS)
        text_location = (_LEFT_MARGIN, _ROW_SIZE)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        # Show the time per inference.
        time_per_infer_text = 'Time per inference: {0}ms'.format(
            int(time_per_infer * 1000))
        text_location = (_LEFT_MARGIN, _ROW_SIZE * 2)
        cv2.putText(image, time_per_infer_text, text_location,
                    cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR,
                    _FONT_THICKNESS)
                                    
        cv2.imshow('Driver Alertness Detection System', image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    
    cap.release()
    cv2.destroyAllWindows()
     

def secondModel():
   # Second model Code to be here
   print("test second model")
   
def thirdModel():
   # third model Code to be here
   print("test third model")



# ===========================================================================================================
# Button Configurations

but1=Button(frame, padx = 6, pady = 6, width = 46, bg = 'white',fg = 'black',relief = GROOVE,command = CNNModel, text = 'Front Camera Monitoring',font = ('Constantia 13 bold'))
but1.place(x = 5,y = 80)

but2=Button(frame, padx = 6,pady = 6, width = 46, bg = 'white', fg = 'black', relief = GROOVE, command = secondModel, text = 'Side Camera Monitoring', font = ('Constantia 13 bold'))
but2.place(x = 5,y = 140)

but3=Button(frame, padx = 6, pady = 6, width=46, bg = 'white', fg = 'black', relief = GROOVE, command = thirdModel, text = 'Thrid Model', font = ('Constantia 13 bold'))
but3.place(x = 5,y = 200)

root.mainloop()

# ===========================================================================================================
