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


def test_interpolation():
    from time import time
    from time import sleep
    x = cv2.imread(r'C:\inz\fhd.jpg', cv2.IMREAD_GRAYSCALE)

    st = time()
    for i in range(100):
        linear = cv2.resize(x, (640, 368), interpolation=cv2.INTER_LINEAR)
    et = time()
    print(f'linear : {(et - st):.64f}')
    cv2.imshow('x', linear)
    cv2.waitKey(0)

    sleep(0.1)

    st = time()
    for i in range(100):
        area = cv2.resize(x, (640, 368), interpolation=cv2.INTER_AREA)
    et = time()
    print(f'area   : {(et - st):.64f}')
    cv2.imshow('x', area)
    cv2.waitKey(0)

    sleep(0.1)

    st = time()
    for i in range(100):
        nearest = cv2.resize(x, (640, 368), interpolation=cv2.INTER_NEAREST)
    et = time()
    print(f'nearest: {(et - st):.64f}')
    cv2.imshow('x', nearest)
    cv2.waitKey(0)
    pass


def test_loss():
    # y_true = [
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ]
    # y_pred = [
    #     [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ]
    # y_pred = [
    #     [0.5, 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0.5, 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
    #     [0.5, 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ]

    # y_true = [
    #     [0.25, 0.50, 0.25],
    #     [0.50, 1.00, 0.50],
    #     [0.25, 0.50, 0.25]
    # ]
    # y_pred = [
    #     [0.10, 0.40, 0.10],
    #     [0.40, 1.00, 0.40],
    #     [0.10, 0.40, 0.10]
    # ]

    # y_true = [[1.0, 0.03, 0.25, 0.5, 0.8]]
    y_true = [[1.0, 0.0, 0.0, 0.0, 0.0]]
    y_pred = [[0.0, 0.0, 0.3, 0.2, 0.6]]

    # y_true = [[0.0]]
    # y_pred = [[1.0]]

    # y_true = [
    #     [
    #         [
    #             [1.0, 0.0],
    #             [0.0, 0.0]
    #         ],
    #         [
    #             [1.0, 0.0],
    #             [0.0, 0.0]
    #         ],
    #     ],
    # ]
    #
    # y_pred = [
    #     [
    #         [
    #             [0.0, 0.0],
    #             [0.0, 0.0]
    #         ],
    #         [
    #             [1.0, 0.0],
    #             [0.0, 0.0]
    #         ],
    #     ],
    # ]

    # y_true = [[0. for _ in range(1000)]]
    # y_true[0][50] = 1.0
    # y_pred = [[0. for _ in range(1000)]]
    # y_pred[0][50] = 0.5

    p(BinaryFocalCrossEntropy()(y_true, y_pred).numpy())
    p(MeanAbsoluteLogError()(y_true, y_pred).numpy())
    p(SumSquaredError()(y_true, y_pred).numpy())
    p(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())


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


def lab_forward(model, x, model_type='h5', input_shape=(0, 0), output_shape=(0, 0), img_channels=1):
    import old
    raw_width, raw_height = x.shape[1], x.shape[0]
    if old.img_channels == 1:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = old.resize(x, (input_shape[1], input_shape[0]))

    y = []
    if model_type == 'h5':
        x = np.asarray(x).reshape((1, input_shape[0], input_shape[1], img_channels)).astype('float32') / 255.0
        y = model.predict(img=x, batch_size=1)[0]
        y = np.moveaxis(y, -1, 0)
    elif model_type == 'pb':
        x = np.asarray(x).reshape((1, img_channels, input_shape[0], input_shape[1])).astype('float32') / 255.0
        model.setInput(x)
        y = model.forward()[0]

    res = []
    box_count = 0
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            confidence = y[0][i][j]
            if confidence < old.confidence_threshold:
                continue
            cx = y[1][i][j]
            cx_f = j / float(output_shape[1])
            cx_f += 1 / float(output_shape[1]) * cx
            cy = y[2][i][j]
            cy_f = i / float(output_shape[0])
            cy_f += 1 / float(output_shape[0]) * cy
            w = y[3][i][j]
            h = y[4][i][j]

            x_min_f = cx_f - w / 2.0
            y_min_f = cy_f - h / 2.0
            x_max_f = cx_f + w / 2.0
            y_max_f = cy_f + h / 2.0

            # if input_shape[1] == 640:
            #     from random import uniform
            #     padding_ratio = uniform(0.0, 0.15)
            #     x_min_f -= (x_max_f - x_min_f) * padding_ratio
            #     x_max_f += (x_max_f - x_min_f) * padding_ratio
            #     y_min_f -= (y_max_f - y_min_f) * padding_ratio
            #     y_max_f += (y_max_f - y_min_f) * padding_ratio

            if x_min_f < 0.0:
                x_min_f = 0.0
            if y_min_f < 0.0:
                y_min_f = 0.0
            if x_max_f < 0.0:
                x_max_f = 0.0
            if y_max_f < 0.0:
                y_max_f = 0.0

            x_min = int(x_min_f * raw_width)
            y_min = int(y_min_f * raw_height)
            x_max = int(x_max_f * raw_width)
            y_max = int(y_max_f * raw_height)
            class_index = -1
            max_percentage = -1
            for cur_channel_index in range(5, len(y)):
                if max_percentage < y[cur_channel_index][i][j]:
                    class_index = cur_channel_index
                    max_percentage = y[cur_channel_index][i][j]
            res.append({
                'confidence': confidence,
                'bbox': [x_min, y_min, x_max, y_max],
                'class': class_index - 5,
                'discard': False
            })
            box_count += 1
            if box_count >= old.max_num_boxes:
                break
        if box_count >= old.max_num_boxes:
            break

    # nms process
    for i in range(len(res)):
        if res[i]['discard']:
            continue
        for j in range(len(res)):
            if i == j or res[j]['discard']:
                continue
            if old.iou(res[i]['bbox'], res[j]['bbox']) >= old.nms_iou_threshold:
                if res[i]['confidence'] >= res[j]['confidence']:
                    res[j]['discard'] = True

    res_copy = res.copy()
    res = []
    for i in range(len(res_copy)):
        if not res_copy[i]['discard']:
            res.append(res_copy[i])
    return sorted(res, key=lambda __x: __x['bbox'][0])


def test_total_lpr_process():
    import old
    old.class_names.append('license_plate')

    old.freeze('checkpoints/2_yolo_4680_epoch_28_loss_0.006669_val_loss_0.034237.h5')
    lpd = cv2.dnn.readNet('model.pb')

    # yolo.freeze('checkpoints/lcd_100_epoch_1768_loss_0.485955_val_loss_14.891500.h5')
    old.freeze('model.h5')
    lcd = cv2.dnn.readNet('model.pb')

    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (640, 368))

    # cap = cv2.VideoCapture('rtsp://admin:samsungg2b!@samsungg2bcctv.iptime.org:1500/video1s1')
    # cap = cv2.VideoCapture(r'C:\inz\videos\g2b.mp4')
    cap = cv2.VideoCapture(r'C:\inz\videos\truen.mkv')
    # cap = cv2.VideoCapture(r'C:\inz\videos\hc_4k_18_day.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon_not_trained.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (2).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (3).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (4).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (5).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (6).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\night.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\night (2).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\night (3).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\night (4).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\1228_4k_5.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\1228_4k_5_night.mp4')

    # inc = 0
    while True:
        frame_exist, x = cap.read()
        if not frame_exist:
            break
        x = old.resize(x, (640, 368))
        raw_x = x.copy()
        res = lab_forward(lpd, x, model_type='pb', input_shape=(368, 640), output_shape=(46, 80), img_channels=1)
        boxed = old.bounding_box(x, res)
        cv2.imshow('boxed', cv2.resize(boxed, (0, 0), fx=2.0, fy=2.0))

        for i, cur_res in enumerate(res):
            x_min, y_min, x_max, y_max = cur_res['bbox']
            lp = raw_x[y_min:y_max, x_min:x_max]
            lp = old.resize(lp, (192, 96))
            # cv2.imwrite(rf'C:\inz\train_data\character_detection_in_lp\ADDONS\g2b_{inc}.jpg', lp)
            # inc += 1
            res = lab_forward(lcd, lp, model_type='pb', input_shape=(96, 192), output_shape=(12, 24), img_channels=1)
            boxed = old.bounding_box(lp, res)
            cv2.imshow('lp', boxed)
        if ord('q') == cv2.waitKey(1):
            break
    # out.release()
    cap.release()
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


def test_kor():
    from glob import glob
    import shutil as sh
    import os
    name_set = set()
    paths = glob(r'\\Desktop-7d5ic8p\LPR3\site_ltc\*.jpg')
    for path in paths:
        original_path = path
        for c in path:
            if ord(c) > 127:
                path = path.replace(c, '')
        sh.move(original_path, path)


def hist_compare_test():
    from glob import glob
    from tqdm import tqdm
    from time import time as t
    from concurrent.futures.thread import ThreadPoolExecutor

    def __load(__path):
        __img = cv2.imread(__path, cv2.IMREAD_GRAYSCALE)
        __hist = cv2.calcHist([__img], [0], None, [256], [0, 255])
        __hist = np.asarray(__hist).reshape(-1)
        __time = int(path.replace('\\', '/').split('/')[-1].split('_')[2])
        return __path, __hist, __time

    pool = ThreadPoolExecutor(16)

    st = t()
    fs = []
    for path in sorted(glob(r'\\Desktop-7d5ic8p\LPR3\2021_02_26\ltc\BLACK_ONE_LINE_7\*.jpg')):
        fs.append(pool.submit(__load, path))

    paths = []
    hists = []
    times = []
    for f in tqdm(fs):
        path, hist, time = f.result()
        paths.append(path)
        hists.append(hist)
        times.append(time)

    duplicated_path_set = set()
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i == j:
                continue

            if abs(times[i] - times[j]) > 30:
                continue

            score = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL)
            if score > 0.95:
                duplicated_path_set.add(paths[i])
                duplicated_path_set.add(paths[j])
                # img_i = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
                # img_j = cv2.imread(paths[j], cv2.IMREAD_GRAYSCALE)
                # cv2.imshow('img_i', cv2.resize(img_i, (512, 256)))
                # cv2.imshow('img_j', cv2.resize(img_j, (512, 256)))
                # cv2.waitKey(0)

    duplicated_paths = list(duplicated_path_set)
    for path in duplicated_paths:
        pass

    et = t()
    print(et - st)


if __name__ == '__main__':
    # compress_test()f
    # test_loss()
    # bounding_box_test()
    # test_interpolation()

    # ccl()
    # cv2_load_test()
    hist_compare_test()
