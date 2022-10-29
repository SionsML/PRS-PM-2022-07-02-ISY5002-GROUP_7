import face_recognition
# import glob
import cv2

#internal import
import constant


# known_face_folder_path = constant.face_storage_dir
# known_face_paths = glob.glob(f"{known_face_folder_path}/*.jpg", recursive=True)
# known_face_encodings = []

def process_frame(frame, rotate=False):
    frame = frame[:, :, ::-1]    # Convert the captured frame from BGR (default in OpenCV) to RGB (needs to be experimented with).
    scale_factor = 1000/frame.shape[0]
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)    # Resize the frame (needs to be experimented with).
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_180)    # Rotate the frame (needs to be experimented with).
    return frame


# for kf_path in known_face_paths:

#     img = cv2.imread(kf_path)
#     img = process_frame(img)
#     face_locations = face_recognition.face_locations(img)
#     img_face_encoding = face_recognition.face_encodings(img, face_locations)
#     if len(img_face_encoding) > 0:
#         known_face_encodings.append(img_face_encoding[0])
    

def perform_face_recognition(frame, known_face_encodings, return_face_location=False):

    # frame_original = frame.copy()
    frame = process_frame(frame, rotate=True)

    face_locations = face_recognition.face_locations(frame)
    print(face_locations)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for i, face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if sum(matches) >= (0.5 * len(known_face_encodings)):
            if return_face_location:
                return True, face_locations[i]
            else:
                return True

    if return_face_location:
        return False, None
    else:
        return False