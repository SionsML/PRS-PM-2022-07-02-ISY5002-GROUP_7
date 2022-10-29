import os
import numpy as np
import tensorflow as tf
import cv2

#internal libraries
import constant
import faceRecognation

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20


# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["Alert", "Drowsy"]


ConvLSTM_model = tf.keras.models.load_model(constant.model_convLSTM)
# Visualization parameters
_ROW_SIZE = 24  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 2
_FONT_THICKNESS = 2

def predict_with_ConvLSTM_model(success,frame):

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
    
    #display classification
    display_text = 'Action Predicted: ' + predicted_class_name + ' with Confidence: %.2f%%' % (predicted_labels_probabilities[predicted_label]*100)
    text_location = (_LEFT_MARGIN, _ROW_SIZE * 2)
    cv2.putText(frame, display_text, text_location,
                        cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR,
                        _FONT_THICKNESS)
    cv2.imshow("frame",frame)
    
    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
# Start capturing video through camera.
def main(known_face_encodings):
    if constant.is_using_camera:
        video_reader = cv2.VideoCapture(0)
    else:
        video_reader = cv2.VideoCapture(constant.sample_video_lstm)

    while True:
        success, frame = video_reader.read()

        # Mirror the image
        if not constant.is_using_camera:
            frame = cv2.flip(frame, 0) #-1 the actual

        #authenticate whether driver face is within the frame
        if constant.is_using_camera:
            is_driver_exists = faceRecognation.perform_face_recognition(frame,known_face_encodings)
        else:
             is_driver_exists = True           
        # cv2.imshow("frame",frame)
        if is_driver_exists:
            predict_with_ConvLSTM_model(success,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    #Release everything
    video_reader.release()
    cv2.destroyAllWindows()