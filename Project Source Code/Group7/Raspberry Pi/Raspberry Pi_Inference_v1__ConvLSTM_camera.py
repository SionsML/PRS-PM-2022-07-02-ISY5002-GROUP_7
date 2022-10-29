import os
import numpy as np
import tensorflow as tf
import cv2

from gpiozero import LED
from time import sleep


# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20


# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["Alert", "Drowsy"]


ConvLSTM_model = tf.keras.models.load_model('20_frames_convlstm_model___Date_Time_2022_10_25__14_58_51___Loss_0.2989948093891144___Accuracy_0.9318181872367859.h5')

def flash_red_light(num_times=3, color="red"):
        for round in range(num_times):
            if color=="red":
                red.on()
                sleep(0.1)
                red.off()
                sleep(0.1)
            elif color == "green":
                green.on()
                sleep(0.1)
                green.off()
                
                


def predict_with_ConvLSTM_model():

    # Declare a list to store video frames we will extract.
    frames = []
    
    # Iterating frames and skipping 10 frames.
    for frame_counter in range(10*SEQUENCE_LENGTH):

        if frame_counter%10==0:
            # Check if frame is not read properly then break the loop.
            if not success:
                break

            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Normalize the resized frame
            normalized_frame = resized_frame / 255

            # Appending the pre-processed frame into the frames list
            frames.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = ConvLSTM_model.predict(np.expand_dims(frames, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    print(f'Driver State Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
    
    return predicted_class_name
    
# Start capturing video through camera.
video_reader = cv2.VideoCapture(0)

# Initial setup and test of LED light
green = LED(25)
red = LED(23)
flash_red_light(num_times=3, color="red")

while True:
    success, frame = video_reader.read()
    cv2.imshow("frame",frame)
    class_name = predict_with_ConvLSTM_model()

    if class_name == "Drowsy":
        print("wake up!")
        flash_red_light(num_times=3, color="red")

    #print(class_name)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#Release everything
video_reader.release()
cv2.destroyAllWindows()