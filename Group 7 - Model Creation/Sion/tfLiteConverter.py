import tensorflow as tf

model = tf.keras.models.load_model('best_model_test.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("best_model_test.tflite", "wb").write(tflite_model)