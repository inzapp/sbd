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
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from time import sleep

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util import ModelUtil


class YoloDataGenerator:
    def __init__(self, image_paths, input_shape, output_shape, batch_size):
        """
        :param input_shape:
            (height, width, channel) format of model input size
            If the channel is 1, train with a gray image, otherwise train with a color image.

        :param output_shape:
            Output shape extracted from built model.

        :param batch_size:
            Batch size of training.
        """
        self.generator_flow = GeneratorFlow(image_paths, input_shape, output_shape, batch_size)

    @classmethod
    def empty(cls):
        """
        Empty class method for only initializing.
        """
        return cls.__new__(cls)

    def flow(self):
        """
        Flow function to load and return the batch.
        """
        return self.generator_flow


class GeneratorFlow(tf.keras.utils.Sequence):
    """
    Custom data generator flow for YOLO model.
    Usage:
        generator_flow = GeneratorFlow(image_paths=image_paths)
    """
    def __init__(self, image_paths, input_shape, output_shapes, batch_size):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.input_width, self.input_height, self.input_channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        self.output_shapes = output_shapes
        if type(self.output_shapes) == tuple:
            self.output_shapes = [self.output_shapes]
        self.batch_size = batch_size
        self.num_output_layers = len(self.output_shapes)
        self.batch_index = 0
        self.pool = ThreadPoolExecutor(8)

        # self.train_type = 'one_layer'
        # from yolo import Yolo
        # if len(Yolo.g_use_layers) > 0:
        #     self.train_type = 'all_layer'

    def __len__(self):
        """
        Number of total iteration.
        """
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def load_label(self, label_path):
        with open(label_path, 'rt') as f:
            lines = f.readlines()
        return lines, label_path

    def is_invalid_label(self, path, label, num_classes):
        class_index, cx, cy, w, h = label
        if class_index < 0 or class_index >= num_classes:
            print(f'\ninvalid class index {int(class_index)} in num_classs {num_classes} : [{path}]')
            return True
        elif cx < 0.0 or cx >= 1.0 or cy < 0.0 or cy >= 1.0:
            print(f'\ninvalid cx or cy. cx : {cx:.6f}, cy : {cy:.6f} : [{path}]')
            return True
        elif w < 0.0 or w > 1.0 or h < 0.0 or h > 1.0:
            print(f'\ninvalid width or height. width : {w:.6f}, height : {h:.6f} : [{path}]')
            return True
        else:
            return False

    def check_invalid_label(self):
        num_classes = self.output_shapes[0][-1] - 5
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, f'{path[:-4]}.txt'))
        invalid_label_paths = set()
        for f in tqdm(fs):
            lines, label_path = f.result()
            for line in lines:
                class_index, cx, cy, w, h = list(map(float, line.split()))
                if self.is_invalid_label(label_path, [class_index, cx, cy, w, h], num_classes):
                    invalid_label_paths.add(label_path)
        if len(invalid_label_paths) > 0:
            print()
            for label_path in list(invalid_label_paths):
                print(label_path)
            print(f'\n{len(invalid_label_paths)} invalid label exists fix it')
            exit(0)
        print('invalid label not found')

    def print_not_trained_box_count(self):
        num_classes = self.output_shapes[0][-1] - 5
        box_count_in_real_data = 0
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, f'{path[:-4]}.txt'))
        invalid_label_paths = set()
        for f in tqdm(fs):
            lines, _ = f.result()
            box_count_in_real_data += len(self.convert_to_boxes(lines))

        image_data_format = tf.keras.backend.image_data_format()
        y_true_obj_count = 0  # obj count in train tensor(y_true)
        for _, batch_y in tqdm(self):
            if self.num_output_layers == 1:
                batch_y = [batch_y]
            max_area = 0
            max_area_index = -1
            for i in range(self.num_output_layers):
                if image_data_format == 'channels_first':
                    cur_area = np.sum(np.ones_like(batch_y[i][:, 0, :, :], dtype=np.float32))
                else:
                    cur_area = np.sum(np.ones_like(batch_y[i][:, :, :, 0], dtype=np.float32))
                if cur_area > max_area:
                    max_area = cur_area
                    max_area_index = i
            if image_data_format == 'channels_first':
                y_true_obj_count += np.sum(batch_y[max_area_index][:, 0, :, :])
            else:
                y_true_obj_count += np.sum(batch_y[max_area_index][:, :, :, 0])
        y_true_obj_count = int(y_true_obj_count)
        not_trained_obj_count = box_count_in_real_data - y_true_obj_count
        not_trained_obj_rate = not_trained_obj_count / box_count_in_real_data * 100.0
        print(f'ground truth obj count : {box_count_in_real_data}')
        print(f'train tensor obj count : {y_true_obj_count}')
        print(f'not trained  obj count : {not_trained_obj_count} ({not_trained_obj_rate:.2f}%)')

    def convert_to_boxes(self, label_lines):
        def get_same_box_index(boxes, cx, cy, w, h):
            box_str = f'{cx}_{cy}_{w}_{h}'
            for i in range(len(boxes)):
                box_cx, box_cy, box_w, box_h = boxes[i]['cx'], boxes[i]['cy'], boxes[i]['w'], boxes[i]['h']
                cur_box_str = f'{box_cx}_{box_cy}_{box_w}_{box_h}'
                if cur_box_str == box_str:
                    return i
            return -1

        boxes = []
        for line in label_lines:
            class_index, cx, cy, w, h = list(map(float, line.split(' ')))
            class_index = int(class_index)
            same_box_index = get_same_box_index(boxes, cx, cy, w, h)
            if same_box_index == -1:
                boxes.append({
                    'class_indexes': [class_index],
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h,
                    'area': w * h})
            elif not class_index in boxes[same_box_index]['class_indexes']:
                boxes[same_box_index]['class_indexes'].append(class_index)
        return boxes

    def sort_middle_last(self, big_last_boxes, small_last_boxes):
        middle_index = int(len(big_last_boxes) / 2)
        middle_last_boxes = []
        for i in range(middle_index):
            middle_last_boxes.append(big_last_boxes[i])
            middle_last_boxes.append(small_last_boxes[i])
        if len(big_last_boxes) % 2 == 1:
            middle_last_boxes.append(small_last_boxes[middle_index])
        return middle_last_boxes
            
    def __getitem__(self, index):
        while True:
            fs, batch_x, batch_y = [], [], []
            for i in range(self.num_output_layers):
                batch_y.append([])
            for path in self.get_next_batch_image_paths():
                fs.append(self.pool.submit(ModelUtil.load_img, path, self.input_channel))
            for f in fs:
                img, _, cur_img_path = f.result()
                img = self.random_blur(img)
                img = ModelUtil.resize(img, (self.input_width, self.input_height))
                x = ModelUtil.preprocess(img)
                batch_x.append(x)

                with open(f'{cur_img_path[:-4]}.txt', mode='rt') as file:
                    label_lines = file.readlines()
                boxes = self.convert_to_boxes(label_lines)
                np.random.shuffle(boxes)

                y = []
                for i in range(self.num_output_layers):
                    y.append(np.zeros(shape=tuple(self.output_shapes[i][1:]), dtype=np.float32))

                layer_mapping_boxes = None
                if self.num_output_layers == 1:
                    layer_mapping_boxes = [boxes]
                else:
                    big_last_boxes = sorted(boxes, key=lambda __x: __x['area'], reverse=False)
                    small_last_boxes = sorted(boxes, key=lambda __x: __x['area'], reverse=True)
                    middle_last_boxes = self.sort_middle_last(big_last_boxes, small_last_boxes)
                    if self.num_output_layers == 2:
                        layer_mapping_boxes = [big_last_boxes, middle_last_boxes]
                    else:
                        layer_mapping_boxes = [big_last_boxes, middle_last_boxes, small_last_boxes]
                        if self.num_output_layers > 3:
                            for i in range(self.num_output_layers - 3):
                                layer_mapping_boxes.append(small_last_boxes)

                image_data_format = tf.keras.backend.image_data_format()
                for i in range(self.num_output_layers):
                    output_rows = float(self.output_shapes[i][1])
                    output_cols = float(self.output_shapes[i][2])
                    for b in layer_mapping_boxes[i]:
                        class_indexes, cx, cy, w, h = b['class_indexes'], b['cx'], b['cy'], b['w'], b['h']
                        center_row = int(cy * output_rows)
                        center_col = int(cx * output_cols)
                        if image_data_format == 'channels_first':
                            y[i][0][center_row][center_col] = 1.0
                            y[i][1][center_row][center_col] = (cx - float(center_col) / output_cols) / (1.0 / output_cols)
                            y[i][2][center_row][center_col] = (cy - float(center_row) / output_rows) / (1.0 / output_rows)
                            y[i][3][center_row][center_col] = w
                            y[i][4][center_row][center_col] = h
                            for class_index in class_indexes:
                                y[i][class_index+5][center_row][center_col] = 1.0
                        else:
                            y[i][center_row][center_col][0] = 1.0
                            y[i][center_row][center_col][1] = (cx - float(center_col) / output_cols) / (1.0 / output_cols)
                            y[i][center_row][center_col][2] = (cy - float(center_row) / output_rows) / (1.0 / output_rows)
                            y[i][center_row][center_col][3] = w
                            y[i][center_row][center_col][4] = h
                            for class_index in class_indexes:
                                y[i][center_row][center_col][class_index+5] = 1.0

                for i in range(self.num_output_layers):
                    batch_y[i].append(y[i])
            batch_x = np.asarray(batch_x).astype('float32')
            for i in range(self.num_output_layers):
                batch_y[i] = np.asarray(batch_y[i]).astype('float32')
            return batch_x, batch_y if self.num_output_layers > 1 else batch_y[0]

    def get_next_batch_image_paths(self):
        start_index = self.batch_size * self.batch_index
        end_index = start_index + self.batch_size
        batch_image_paths = self.image_paths[start_index:end_index]
        self.batch_index += 1
        if self.batch_index == self.__len__():
            self.batch_index = 0
            np.random.shuffle(self.image_paths)
        return batch_image_paths

    def random_blur(self, img):
        if np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            else:
                img = cv2.blur(img, (2, 2))
        return img

    def random_adjust(self, img):
        adjust_opts = ['saturation', 'brightness', 'contrast', 'noise']
        np.random.shuffle(adjust_opts)
        for i in range(len(adjust_opts)):
            img = self.adjust(img, adjust_opts[i])
        return img

    def adjust(self, img, adjust_type):
        weight = np.random.uniform(0.75, 1.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)

        if adjust_type == 'saturation':
            s = np.asarray(s).astype('float32') * weight
            s = np.clip(s, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'brightness':
            v = np.asarray(v).astype('float32') * weight
            v = np.clip(v, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'contrast':
            weight = np.random.uniform(0.0, 0.25)
            criteria = np.random.uniform(84.0, 170.0)
            v = np.asarray(v).astype('float32')
            v += (criteria - v) * weight
            v = np.clip(v, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'noise':
            range_min = np.random.uniform(0.0, 25.0)
            range_max = np.random.uniform(0.0, 25.0)
            v = np.asarray(v).astype('float32')
            v += np.random.uniform(-range_min, range_max, size=v.shape)
            v = np.clip(v, 0.0, 255.0).astype('uint8')

        img = cv2.merge([h, s, v])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

