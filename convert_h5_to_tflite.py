import os
import tensorflow as tf

with tf.device('/cpu:0'):
    os.system('ls -alrt *.h5')
    model_name = input('paste model name : ')
    model = tf.keras.models.load_model(model_name, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_name = f'{model_name[:-3]}.tflite'
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)
    print(f'\nmodel conversion success. save to {tflite_model_name}')

