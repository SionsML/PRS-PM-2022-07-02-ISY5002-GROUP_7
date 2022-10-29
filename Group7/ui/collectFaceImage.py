import cv2
import os
import glob
import face_recognition
import pickle
import shutil

#internal import
import constant

# Visualization parameters
_ROW_SIZE = 24  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1

def collectFaceImageAndStore():

    for filename in os.listdir(constant.face_storage_dir):
        file_path = os.path.join(constant.face_storage_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    cam = cv2.VideoCapture(0)
    #cam.set(CAP_PROP_WB_TEMPERATURE, 4500)


    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    face_detector = cv2.CascadeClassifier(constant.face_detector_xml_dir)
    # For each person, enter one numeric face id
    face_id = 1#input('\n enter user id end press <return> ==>  ')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while(True):
        ret, img = cam.read()
        #img = cv2.flip(img, -1) # flip video image vertically
        text_location = (_LEFT_MARGIN, _ROW_SIZE)
        cv2.putText(img, 'Please turn your face left and right! ', text_location,
                    cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR,
                    _FONT_THICKNESS)       
        text_location = (_LEFT_MARGIN, _ROW_SIZE * 2)
        cv2.putText(img, 'Please ensure your face within the blue box!!', text_location,
                    cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR,
                    _FONT_THICKNESS)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite(constant.face_storage_path + str(face_id) + '.' +  
                        str(count) + ".jpg", cv2.cvtColor(gray, cv2.COLOR_RGB2BGR))
            cv2.imshow('image', img)
            cv2.waitKey(100)
        k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            keepAndTrain()
            break
        elif count >= 30: # Take 30 face sample and stop video
            keepAndTrain()
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def process_frame(frame, rotate=False):
    frame = frame[:, :, ::-1]    # Convert the captured frame from BGR (default in OpenCV) to RGB (needs to be experimented with).
    scale_factor = 1000/frame.shape[0]
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)    # Resize the frame (needs to be experimented with).
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_180)    # Rotate the frame (needs to be experimented with).
    return frame

def keepAndTrain():
    known_face_folder_path = constant.face_storage_dir
    known_face_paths = glob.glob(f"{known_face_folder_path}/*.jpg", recursive=True)
    known_face_encodings = []

    for kf_path in known_face_paths:

        img = cv2.imread(kf_path)
        img = process_frame(img)
        face_locations = face_recognition.face_locations(img)
        img_face_encoding = face_recognition.face_encodings(img, face_locations)
        if len(img_face_encoding) > 0:
            known_face_encodings.append(img_face_encoding[0])

    with open(constant.model_face_recognition, 'wb') as f:
        pickle.dump(known_face_encodings, f)