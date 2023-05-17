"""
Authors : inzapp

Github url : https://github.com/inzapp/sbd

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
import cv2
import numpy as np


class Util:
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
    def load_img(path, channel):
        color_mode = cv2.IMREAD_COLOR if channel == 3 else cv2.IMREAD_GRAYSCALE
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), color_mode)
        raw_target = img
        if color_mode == cv2.IMREAD_COLOR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        return img, raw_target, path

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
    def nms(boxes, nms_iou_threshold):
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        for i in range(len(boxes) - 1):
            if boxes[i]['discard']:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j]['discard'] or boxes[i]['class'] != boxes[j]['class']:
                    continue
                if Util.iou(boxes[i]['bbox_norm'], boxes[j]['bbox_norm']) > nms_iou_threshold:
                    boxes[j]['discard'] = True

        y_pred_copy = np.asarray(boxes.copy())
        boxes = []
        for i in range(len(y_pred_copy)):
            if not y_pred_copy[i]['discard']:
                boxes.append(y_pred_copy[i])
        return boxes

    @staticmethod
    def nms(boxes, nms_iou_threshold):
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        for i in range(len(boxes) - 1):
            if boxes[i]['discard']:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j]['discard'] or boxes[i]['class'] != boxes[j]['class']:
                    continue
                if Util.iou(boxes[i]['bbox_norm'], boxes[j]['bbox_norm']) > nms_iou_threshold:
                    boxes[j]['discard'] = True

        y_pred_copy = np.asarray(boxes.copy())
        boxes = []
        for i in range(len(y_pred_copy)):
            if not y_pred_copy[i]['discard']:
                boxes.append(y_pred_copy[i])
        return boxes

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

