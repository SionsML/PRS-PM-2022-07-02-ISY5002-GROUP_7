def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]



import numpy as np
import cv2
import datetime
#from playsound import playsound
import pygame


#internal library
import constant
import faceRecognation

#set music
pygame.mixer.init()

def classify_tf(img,width,height):
    # Load the TFLite model and allocate tensors.
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=constant.model_alertness_detection)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    img = cv2.resize(img,(width,height),3)


    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = np.array(img, dtype=np.float32)
    input_tensor= np.array(np.expand_dims(input_data,0))

    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    top_k = np.argmax(results)
    #print(results)
    return results

def classify_tflite(img,width,height):
    # Load the TFLite model and allocate tensors.
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=constant.model_alertness_detection)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    img = cv2.resize(img,(width,height),3)


    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = np.array(img, dtype=np.float32)
    input_tensor= np.array(np.expand_dims(input_data,0))

    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    top_k = np.argmax(results)
    #print(results)
    return results


#import argparse
import sys
import time

import cv2


# Visualization parameters
_ROW_SIZE = 24  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 2
_FONT_THICKNESS = 2
_MODEL_FPS = 5  # Ensure the input images are fed to the model at this fps.
_MODEL_FPS_ERROR_RANGE = 0.1  # Acceptable error range in fps.



"""
    Method to continuous capture and classify are referenced/adapted from:
    Tensorflow Video Classification on Mobile & Edge device. 
    https://www.tensorflow.org/lite/examples/video_classification/overview
""" 

def classifyVideo(camera_id: int, width: int, height: int,known_face_encodings) -> None:


  # Variables to calculate FPS
    counter, fps, last_inference_start_time, time_per_infer = 0, 0, 0, 0
    categories = []

  # Start capturing video input from the camera
    #cap = cv2.VideoCapture(cv2.CAP_V4L2)#camera_id
    if constant.is_using_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(constant.sample_video_cnn)

    #cap = cv2.VideoCapture('IMG_1706.MOV')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    labels = load_labels(constant.label_alertness_detection)

    accuracy_value = 0
    top_k = 0
    classification = 'Driver not identified'

    classification_cnt = 0
    pre_classification = 'normaldriving'

  # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
              'ERROR: Unable to read from webcam. Please verify your webcam settings.'
          )
        counter += 1

        # Mirror the image
        if not constant.is_using_camera:
            image = cv2.flip(image, 0) #-1 the actual

        # Ensure that frames are feed to the model at {_MODEL_FPS} frames per second
        # as required in the model specs.
        current_frame_start_time = time.time()
        diff = current_frame_start_time - last_inference_start_time
        #authenticate whether driver face is within the frame it's not within the scope of side camera
        #is_driver_exists = faceRecognation.perform_face_recognition(image,known_face_encodings)
        is_driver_exists = True
 
        

        if diff * _MODEL_FPS >= (1 - _MODEL_FPS_ERROR_RANGE) and is_driver_exists:
            # Store the time when inference starts.
            last_inference_start_time = current_frame_start_time

            # Calculate the inference FPS
            fps = 1.0 / diff

            # Convert the frame to RGB as required by the TFLite model.
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #print('Here')
            # Feed the frame to the video classification model.
            if constant.deployment_type == 'tf':
                categories = classify_tf(frame_rgb,width,height)
            else:
                categories = classify_tflite(frame_rgb,width,height)

            top_k = np.argmax(categories)
            
            accuracy_value = categories[top_k]
            classification = labels[top_k]

            print(str(top_k) + ' ---- ' + classification)

            # Calculate time required per inference.
            time_per_infer = time.time() - current_frame_start_time


        if is_driver_exists:
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
            
                
            accuracy = 'Accurcy : %.2f%%' % (accuracy_value*100) #50#categories[top_k]*100            
            result_text = classification + ' (' + str(accuracy) + ')'

            if classification_cnt > 30 and (accuracy_value*100)> constant.model_cnn_accuracy_threshold and pre_classification != 'normaldriving':
                pygame.mixer.music.load(constant.sound_focus_alert)
                pygame.mixer.music.play(1)    
                classification_cnt = 0   

            if classification == pre_classification:
                classification_cnt +=1
            else:
                classification_cnt = 0

            pre_classification = classification
            if constant.is_for_testing and (accuracy_value*100)> constant.model_cnn_accuracy_threshold:
                cv2.imwrite(constant.model_result_model_alertness_storage_dir + constant.model_alertness_detection.split('.')[0] + '-' +  
                        result_text + '-' + str(datetime.datetime.now()) + ".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        else:
            result_text = classification
        # Skip the first 2 lines occupied by the fps and time per inference.
        text_location = (_LEFT_MARGIN, (0 + 3) * _ROW_SIZE)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                  _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)



        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('video_classification', image)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def alertnessDetection(known_face_encodings):
  classifyVideo(int(1), 224, 224,known_face_encodings)

