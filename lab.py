"""
Authors : inzapp

Github url : https://github.com/inzapp/c-yolo

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import tensorflow as tf
from cv2 import cv2


def main():
    # model = tf.keras.models.load_model('checkpoints/model_epoch_1_loss_0.0186_val_loss_0.0157.h5')
    model = tf.keras.models.load_model('model.h5')
    # model = tf.keras.models.load_model('lp_type_model.h5')
    previous_channel = -1
    total_convolution_count = 0
    for layer in model.layers:
        if str(layer).lower().find('conv') == -1:
            continue
        if previous_channel == -1:
            previous_channel = layer.__input_shape[3]
        shape = layer.__output_channel
        if type(shape) is list:
            shape = shape[0]
        h, w, c = shape[1:]
        d = tf.keras.utils.serialize_keras_object(layer)
        kernel_size = d['config']['kernel_size']
        strides = d['config']['strides']
        cur_convolution_count = h * w * c * previous_channel
        cur_convolution_count = cur_convolution_count * kernel_size[0] * kernel_size[1]
        cur_convolution_count = cur_convolution_count / (strides[0] * strides[1])
        total_convolution_count += cur_convolution_count
        previous_channel = c
    print(total_convolution_count)


def p(v):
    try:
        len(v)
        for c in v:
            print(f'{c:.4f} ', end='')
        print()
    except TypeError:
        print(f'{v:.4f}')


class SumOverLogError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mae = tf.math.abs(tf.math.subtract(y_pred, y_true))
        sub = tf.math.subtract(tf.constant(1.0 + 1e-7), mae)
        log = -tf.math.log(sub)
        return tf.keras.backend.mean(tf.keras.backend.sum(log, axis=-1))


class SumSquaredError(tf.keras.losses.Loss):
    def __init__(self, coord=5.0):
        self.coord = coord
        super().__init__()

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_sum(tf.math.square(y_pred - y_true)) * self.coord


class FalsePositiveWeightedError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        fp = tf.keras.backend.clip(-(y_true - y_pred), 0.0, 1.0)
        fp = -tf.math.log(1.0 + 1e-7 - fp)
        fp = tf.keras.backend.mean(tf.keras.backend.sum(fp, axis=-1))
        loss = tf.math.abs(y_pred - y_true)
        loss = -tf.math.log(1.0 + 1e-7 - loss)
        loss = tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
        return loss + fp


class MeanAbsoluteLogError(tf.keras.losses.Loss):
    """
    False positive weighted loss function.
    f(x) = -log(1 - MAE(x))
    Usage:
     model.compile(loss=[MeanAbsoluteLogError()], optimizer="sgd")
    """

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = tf.math.abs(y_pred - y_true)
        loss = -tf.math.log(1.0 + 1e-7 - loss)
        loss = tf.keras.backend.mean(loss)
        return loss


class BinaryFocalCrossEntropy(tf.keras.losses.Loss):
    """
    Binary form of focal cross entropy.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[BinaryFocalCrossEntropy(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer="adam")
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
        super().__init__()

    def call(self, y_true, y_pred):
        from keras import backend as K
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * self.alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), self.gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))


class SSEYoloLoss(tf.keras.losses.Loss):
    def __init__(self, coord=5.0):
        self.coord = coord
        super(SSEYoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        confidence_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0]))
        xy_loss = tf.reduce_sum(tf.reduce_sum(tf.square(y_true[:, :, :, 1:3] - y_pred[:, :, :, 1:3]), axis=-1) * y_true[:, :, :, 0])
        wh_true = tf.sqrt(y_true[:, :, :, 3:5] + 1e-4)
        wh_pred = tf.sqrt(y_pred[:, :, :, 3:5] + 1e-4)
        wh_loss = tf.reduce_sum(tf.reduce_sum(tf.square(wh_true - wh_pred), axis=-1) * y_true[:, :, :, 0])
        classification_loss = tf.reduce_sum(tf.reduce_sum(tf.square(y_true[:, :, :, 5:] - y_pred[:, :, :, 5:]), axis=-1) * y_true[:, :, :, 0])
        return confidence_loss + (xy_loss * self.coord) + (wh_loss * self.coord) + classification_loss


def interpolation_test():
    x = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.0, 0.4, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.0, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    )
    x = np.clip(x, 0.7, 1.0)
    cv2.normalize(x, x, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    x = (x * 255.0).astype('uint8')
    x = cv2.resize(x, (500, 500), interpolation=cv2.INTER_LINEAR)

    for i in range(100):
        x_copy = x.copy()
        threshold = float(i / 100.0)
        _, y = cv2.threshold(x, int(255.0 * threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            for contour in contours:
                x1, y1, w, h = cv2.boundingRect(contour)
                cv2.rectangle(x_copy, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), thickness=2)
        print(threshold)
        cv2.imshow('x', x_copy)
        cv2.imshow('y', y)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_masked_image():
    import os
    from glob import glob
    for path in glob(r'C:\inz\train_data\lp_detection_yolo\crime_day_1f_1\*.jpg'):
        label_path = f'{path[:-4]}.txt'
        if os.path.exists(label_path) and os.path.isfile(label_path):
            with open(label_path, mode='rt') as f:
                lines = f.readlines()
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            raw_height, raw_width = img.shape[0], img.shape[1]
            mask = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                class_index, cx, cy, w, h = list(map(float, line.split(' ')))
                cx = int(cx * raw_width)
                cy = int(cy * raw_height)
                w = int(w * raw_width)
                h = int(h * raw_height)
                x1, y1 = int(cx - w / 2.0), int(cy - h / 2.0)
                x2, y2 = int(cx + w / 2.0), int(cy + h / 2.0)
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        mask[y][x] = img[y][x]
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('img', mask)
            cv2.waitKey(0)


def __iou(a, b):
    """
    Intersection of union function.
    :param a: [x_min, y_min, x_max, y_max] format box a
    :param b: [x_min, y_min, x_max, y_max] format box b
    """
    a_x_min, a_y_min, a_x_max, a_y_max = a
    b_x_min, b_y_min, b_x_max, b_y_max = b
    intersection_width = min(a_x_max, b_x_max) - max(a_x_min, b_x_min)
    intersection_height = min(a_y_max, b_y_max) - max(a_y_min, b_y_min)
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
    b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
    union_area = a_area + b_area - intersection_area
    return intersection_area / (float(union_area) + 1e-5)


def view_boxed_image():
    # 10000
    # rs = [
    #     [464, 306, 29, 10],  # 0.6952
    #     [467, 308, 29, 9],  # 0.9120
    #     [391, 82, 58, 14],  # 0.3584
    #     [153, 92, 60, 14],  # 0.4219
    #     [395, 77, 49, 25],  # 0.9748
    #     [159, 88, 46, 24],  # 0.9833
    #     [453, 298, 57, 26],  # 3404
    # ]

    # 20000
    # rs = [
    #     [464, 306, 30, 9],  # 0.4074
    #     [465, 308, 32, 9],  # 0.8354
    #     [396, 76, 47, 28],  # 0.8911
    #     [159, 88, 47, 26],  # 0.9013
    # ]

    # 30000
    # rs = [
    #     [407, 86, 26, 9],  # 0.2925
    #     [464, 306, 30, 9],  # 0.9347
    #     [365, 308, 32, 9],  # 0.8865
    #     [395, 76, 48, 28],  # 0.8865
    #     [161, 87, 44, 26],  # 0.9150
    # ]

    # 40000
    # rs = [
    #     [406, 86, 24, 8],  # 0.5112
    #     [464, 307, 31, 8],  # 0.3611
    #     [466, 308, 31, 9],  # 0.9661
    #     [395, 76, 48, 28],  # 0.9744
    #     [159, 88, 47, 25],  # 0.9672
    # ]

    # 50000
    rs = [
        [406, 86, 26, 8],  # 0.3468
        [464, 308, 30, 8],  # 0.5850
        [465, 308, 32, 9],  # 0.9766
        [395, 76, 47, 27],  # 0.9554
        [158, 86, 48, 26],  # 0.9592
        [451, 298, 59, 27]  # 0.4542
    ]

    # 60000
    # rs = [
    #     [405, 87, 27, 8],  # 0.3630
    #     [463, 309, 32, 9],  # 0.4090
    #     [466, 309, 30, 9],  # 0.9731
    #     [395, 77, 46, 26],  # 0.9579
    #     [160, 88, 44, 24],  # 0.9695
    #     [454, 297, 53, 27],  # 0.6480
    # ]

    # 70000
    # rs = [
    #     [466, 308, 31, 8],  # 0.9720
    #     [153, 92, 60, 15],  # 0.2956
    #     [396, 76, 46, 27],  # 0.9910
    #     [160, 87, 45, 25],  # 0.9932
    #     [455, 297, 53, 28],  # 0.6696
    # ]

    # 80000
    rs = [
        [466, 309, 32, 8],  # 0.9608
        [152, 91, 61, 16],  # 0.4822
        [396, 77, 45, 27],  # 0.9737
        [160, 87, 45, 24],  # 0.9655
        [455, 298, 53, 27],  # 0.7975
    ]

    # 7000 (anchor reset)
    # rs = [
    #     [470, 309, 23, 9],  # 0.7925
    #     [382, 84, 74, 13],  # 0.3354
    #     [381, 72, 68, 36],  # 0.4766
    #     [385, 74, 67, 32],  # 0.9753
    #     [147, 84, 70, 32],  # 0.9836
    #     [442, 294, 74, 37],  # 0.3725
    #     [442, 292, 79, 35],  # 0.7197
    # ]

    # 10000 (anchor reset)
    # rs = [
    #     [470, 307, 23, 10],  # 0.8034
    #     [383, 73, 70, 31],  # 0.5537
    #     [148, 86, 70, 29],  # 0.7514
    # ]

    # 20000 (anchor reset)
    # rs = [
    #     [470, 308, 23, 10],  # 0.9280
    #     [383, 73, 70, 33],  # 0.9498
    #     [148, 85, 72, 32],  # 0.8932
    #     [439, 293, 86, 36],  # 0.5859
    # ]

    # 30000 (anchor reset)
    # rs = [
    #     [470, 307, 23, 10],  # 0.9485
    #     [384, 72, 69, 36],  # 0.9893
    #     [148, 81, 69, 38],  # 0.9886
    #     [440, 291, 83, 40],  # 0.6671
    # ]

    # 40000 (anchor reset)
    # rs = [
    #     [470, 309, 22, 9],  # 0.9414
    #     [384, 74, 68, 32],  # 0.9901
    #     [149, 87, 69, 29],  # 0.9838
    #     [440, 294, 83, 34],  # 0.9109
    # ]

    # 40000 (anchor reset) compile mode 1
    # rs = [
    #     [470, 309, 22, 9],  # 0.9575
    #     [384, 73, 68, 33],  # 0.9931
    #     [149, 86, 69, 30],  # 0.9893
    #     [440, 294, 83, 34],  # 0.8998
    # ]

    # 70000 (anchor reset)
    # rs = [
    #     [470, 308, 23, 10],  # 0.9847
    #     [383, 73, 70, 34],  # 0.9959
    #     [147, 84, 71, 32],  # 0.9962
    #     [436, 292, 90, 38],  # 0.8391
    # ]

    # 90000 (anchor reset)
    # rs = [
    #     [469, 308, 24, 10],  # 0.9800
    #     [383, 74, 69, 32],  # 0.9970
    #     [148, 85, 70, 30],  # 0.9979
    #     [435, 293, 92, 36]  # 0.7978
    # ]

    # direct caffe forwarding test 90000 (anchor reset)
    # rs = [
    #     [399, 85, 38, 11],  # 1.0
    #     [163, 94, 40, 10],  # 1.0
    #     [455, 304, 50, 12],  # 0.96
    #     [454, 305, 53, 13],  # 0.98
    # ]

    # direct caffe forwarding test 90000 (anchor reset)
    # rs = [
    #     [472, 308, 18, 9],
    #     [394, 83, 48, 14],
    #     [158, 93, 50, 14],
    #     [447, 303, 67, 16]
    # ]

    # raw with prec anchors
    # rs = [
    #     [483, 311, 30, 10],
    #     [396, 77, 45, 27],
    #     [160, 86, 45, 27]
    # ]

    # cfg modified with prev anchors
    # rs = [
    #     [480, 311, 30, 10],
    #     [396, 77, 45, 27],
    #     [160, 86, 45, 27]
    # ]

    rs = [
        [472, 308, 18, 9],
        [394, 83, 48, 14],
        [158, 93, 50, 14],
        [447, 303, 67, 16]
    ]

    img = cv2.imread(r'C:\inz\truen_1.jpg', cv2.IMREAD_COLOR)
    for r in rs:
        x1, y1, width, height = r
        x2 = x1 + width
        y2 = y1 + height
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    img = cv2.resize(img, (1280, 720))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def model_summary():
    model = tf.keras.models.load_model(r'C:\inz\git\yolo-lab\checkpoints\person\new_loss_person_3_class_192_96_epoch_182_val_mAP_0.7454.h5', compile=False)
    model.summary()


def model_forward():
    from yolo import Yolo
    from glob import glob
    model = Yolo(pretrained_model_path=r'C:\inz\git\yolo-lab\checkpoints\sgd_v2_person_info_detector_192_96_epoch_23_val_mAP_0.2522.h5', class_names_file_path=r'X:\person\face_helmet_added\validation\classes.txt')
    # model.predict_images([r'C:\inz\detail_96_192_1.jpg'])
    model.predict_images(glob(r'X:\person\face_helmet_added\validation\*.jpg'))


def color_regression():
    from glob import glob
    for path in glob(r'C:\inz\train_data\car_color_regression\*.jpg'):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        x = np.moveaxis(img, -1, 0)
        x = np.mean(x, axis=-1)
        x = np.mean(x, axis=-1)
        x = x.reshape((1, 1, 3)).astype('uint8')
        x = cv2.resize(x, (128, 128))
        print(path)
        print(x)
        print()
        cv2.imshow('img', img)
        cv2.imshow('x', x)
        cv2.waitKey(0)


if __name__ == '__main__':
    model_summary()
