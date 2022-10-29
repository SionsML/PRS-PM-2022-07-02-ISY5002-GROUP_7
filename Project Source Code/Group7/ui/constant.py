#this is to keep all folder location


face_detector_xml_dir = '../models/haarcascade_frontalface_default.xml'
face_storage_dir = '../face_record'
face_storage_path = '../face_record/user'
face_landmark_dat_dir = '../models/shape_predictor_68_face_landmarks.dat'

sound_doorbell = '../sounds/doorbell.wav'
sound_yawn = '../sounds/yawn.mp3'
sound_focus_alert = '../sounds/focus_alert.mp3'

#model for facial landmark
model_facial_landmark = '../models/best_model_test.tflite'
#model for body posture and it's landmark and it's label
model_alertness_detection = '../models/trxlrn_MobileNetV2_09c.tflite'
label_alertness_detection = '../models/label_trxlrn_MobileNetV2_09c.txt'
#model for rnn
model_lrcn = '../models/LRCN_model_24102022.h5'
model_convLSTM = '../models/ConvLSTM_26102022.h5'
#pickle file for face recognition
model_face_recognition = '../models/known_face.pkl'

model_result_model_alertness_storage_dir = '../cnn_result/'

#indicate confident level threshold being used before system alert in CNN Model
model_cnn_accuracy_threshold = 95

#1. As intended deployment is edge device, we provide switching of using Interpreter from tflite-runtime or from tensorflow. The tester just need to change the value from tf to tflite
deployment_type='tf'
#2. this is intended to capture fram from video that running. The files will be kept in model_result_model_alertness_storage_dir value and filename will indicate the classname and accuracy
is_for_testing =False
#3. This is indicator to differentiate whether to take from live camera or video
is_using_camera = False
#4. this indicator to differentiate device type
device_type = 'laptop' #others for non Raspberry Pi Device
#5. Sample videos being used for testing. This will not be submitted
sample_video_lstm = '../sample_videos/IMG_1706.MOV'
sample_video_lrcn = '../sample_videos/IMG_1706.MOV'
sample_video_facialLandmark = '../sample_videos/IMG_7862.MOV'
sample_video_cnn = '../sample_videos/IMG_7859.MOV'