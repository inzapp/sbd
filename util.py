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
import os
from glob import glob
from time import perf_counter

import cv2
import numpy as np
import tensorflow as tf
from keras_flops import get_flops


class ModelUtil:
    def __init__(self):
        pass

    @staticmethod
    def print_error_exit(msg):
        msg_type = type(msg)
        if msg_type is str:
            msg = [msg]
        msg_type = type(msg)
        if msg_type is list:
            print()
            for s in msg:
                print(f'[ERROR] {s}')
        else:
            print(f'[print_error_exit] msg print failure. invalid msg type : {msg_type}')
        exit(-1)

    @staticmethod
    def available_device():
        devices = tf.config.list_physical_devices()
        for device in devices:
            if device.device_type.lower() == 'gpu':
                return 'gpu'
        return 'cpu'

    @staticmethod
    def init_image_paths(image_path, validation_split=0.0):
        if image_path.endswith('.txt'):
            with open(image_path, 'rt') as f:
                image_paths = f.readlines()
            for i in range(len(image_paths)):
                image_paths[i] = image_paths[i].replace('\n', '')
        else:
            image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        np.random.shuffle(image_paths)
        return image_paths

    @staticmethod
    def get_zero_mod_batch_size(image_paths_length):
        zero_mod_batch_size = 1
        for i in range(1, 256, 1):
            if image_paths_length % i == 0:
                zero_mod_batch_size = i
        return zero_mod_batch_size

    @staticmethod
    def get_gflops(model):
        return get_flops(model, batch_size=1) * 1e-9

    @staticmethod
    def load_img(path, channel):
        color_mode = cv2.IMREAD_COLOR if channel == 3 else cv2.IMREAD_GRAYSCALE
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), color_mode)
        raw_bgr = img
        if color_mode == cv2.IMREAD_COLOR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        return img, raw_bgr, path

    @staticmethod
    def resize(img, size):
        img_h, img_w = img.shape[:2]
        if img_h > size[0] or img_w > size[1]:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def preprocess(img):
        x = np.asarray(img).astype('float32') / 255.0
        if len(img.shape) == 1:
            x = x.reshape(img.shape + (1,))
        return x

    @staticmethod
    def get_width_height_channel_from_input_shape(input_shape):
        if input_shape[0] in [1, 3]:
            return input_shape[2], input_shape[1], input_shape[0]
        elif input_shape[2] in [1, 3]:
            return input_shape[1], input_shape[0], input_shape[2]

    @staticmethod
    @tf.function
    def graph_forward(model, x, device):
        with tf.device(f'/{device}:0'):
            return model(x, training=False)

    @staticmethod
    def check_forwarding_time(model, device):
        input_shape = model.input_shape[1:]
        mul = 1
        for val in input_shape:
            mul *= val

        forward_count = 32
        noise = np.random.uniform(0.0, 1.0, mul * forward_count)
        noise = np.asarray(noise).reshape((forward_count, 1) + input_shape).astype('float32')
        ModelUtil.graph_forward(model, noise[0], device)  # only first forward is slow, skip first forward in check forwarding time

        st = perf_counter()
        for i in range(forward_count):
            ModelUtil.graph_forward(model, noise[i], device)
        et = perf_counter()
        forwarding_time = ((et - st) / forward_count) * 1000.0
        print(f'model forwarding time with {device} : {forwarding_time:.2f} ms')

    @staticmethod
    def nms(boxes, nms_iou_threshold):
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        for i in range(len(boxes) - 1):
            if boxes[i]['discard']:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j]['discard'] or boxes[i]['class'] != boxes[j]['class']:
                    continue
                if ModelUtil.iou(boxes[i]['bbox_norm'], boxes[j]['bbox_norm']) > nms_iou_threshold:
                    boxes[j]['discard'] = True

        y_pred_copy = np.asarray(boxes.copy())
        boxes = []
        for i in range(len(y_pred_copy)):
            if not y_pred_copy[i]['discard']:
                boxes.append(y_pred_copy[i])
        return boxes

    @staticmethod
    def init_class_names(class_names_file_path):
        if os.path.exists(class_names_file_path) and os.path.isfile(class_names_file_path):
            with open(class_names_file_path, 'rt') as classes_file:
                class_names = [s.replace('\n', '') for s in classes_file.readlines()]
                num_classes = len(class_names)
            return class_names, num_classes
        else:
            return [], 0

    @staticmethod
    def iou(a, b):
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

    @staticmethod
    def is_background_color_bright(bgr):
        """
        Determine whether the color is bright or not.
        :param bgr: bgr scalar tuple.
        :return: true if parameter color is bright and false if not.
        """
        tmp = np.zeros((1, 1), dtype=np.uint8)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tmp, (0, 0), (1, 1), bgr, -1)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        return tmp[0][0] > 127

