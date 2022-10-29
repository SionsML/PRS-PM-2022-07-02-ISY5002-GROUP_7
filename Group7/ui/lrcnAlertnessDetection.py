import os
import numpy as np
import tensorflow as tf

import cv2

#internal libraries
import constant

#IMG_SIZE = 480
BATCH_SIZE = 8
EPOCHS = 10

MAX_SEQ_LENGTH = 300
NUM_FEATURES = 2048

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 40


# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["class0", "class1"]


LCRN_model = tf.keras.models.load_model(constant.model_lrcn)


def predict_with_LRCN_model(success,frame):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    

    # Get the width and height of the video.
    #original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    #original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    #video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    #skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(10*SEQUENCE_LENGTH):
        # Set the current frame position of the video.
        #video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        #success, frame = video_reader.read()
        #cv2.imshow("frame",frame) 
        if frame_counter%10==0:
            #print(frame_counter)            
            # Check if frame is not read properly then break the loop.
            if not success:
                break

            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
            normalized_frame = resized_frame / 255

            # Appending the pre-processed frame into the frames list
            frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LCRN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
    # Release the VideoCapture object. 

    #return predicted_class_name


def main():

    if constant.is_using_camera:
        video_reader = cv2.VideoCapture(0)
    else:
        video_reader = cv2.VideoCapture(constant.sample_video_lrcn)

    while True:
        success, frame = video_reader.read()
        # Mirror the image
        frame = cv2.flip(frame, 0) #-1 the actual

        cv2.imshow("frame",frame)
        predict_with_LRCN_model(success,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video_reader.release()
    cv2.destroyAllWindows()