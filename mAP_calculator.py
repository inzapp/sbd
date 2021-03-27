import numpy as np
import tensorflow as tf
from cv2 import cv2

from yolo import Yolo

iou_thresholds = [0.5]
confidence_thresholds = np.asarray(list(range(5, 100, 5))).astype('float32') / 100.0


@tf.function
def predict_on_graph(model, x):
    return model(x, training=False)


def main(model_path, image_paths, class_names_file_path=''):
    global iou_thresholds, confidence_thresholds
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    input_size = (input_shape[1], input_shape[0])
    color_mode = cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR
    num_classes = model.output_shape[-1] - 5

    print(input_shape)
    print(input_size)
    print(color_mode)
    print(num_classes)

    class_precision_sum = [0.0 for _ in range(num_classes)]
    class_recall_sum = [0.0 for _ in range(num_classes)]
    for path in image_paths:
        x = cv2.imread(path, color_mode)
        x = cv2.resize(x, input_size)
        x = np.asarray(x).astype('float32').reshape((1,) + input_shape) / 255.0
        y = np.asarray(predict_on_graph(model, x))
        print(y)


if __name__ == '__main__':
    from glob import glob

    main(
        'loon_detector_model_epoch_184_f1_0.9651_val_f1_0.8532.h5',
        glob('C:/inz/train_data/loon_detection_train/*.jpg'),
        class_names_file_path=r'C:\inz\train_data\loon_detection_train\classes.txt')
