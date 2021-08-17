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
        self.output_shapes = output_shapes
        self.batch_size = batch_size
        self.random_indexes = np.arange(len(self.image_paths))
        self.pool = ThreadPoolExecutor(8)

        queue_size = 64
        self.batch_index = 0
        self.batch_q = Queue(maxsize=queue_size)
        self.insert_thread_running = False
        np.random.shuffle(self.random_indexes)

    def __len__(self):
        """
        Number of total iteration.
        """
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # print()
        # print(self.batch_q.qsize())
        # print()
        return self.batch_q.get(block=True)

    def start(self):
        if self.insert_thread_running:
            print('insert thread is already running !!!')
            return

        self.insert_thread_running = True
        for _ in range(4):
            insert_thread = Thread(target=self.__insert_batch_into_q)
            insert_thread.setDaemon(True)
            insert_thread.start()

    @staticmethod
    def __convert_to_boxes(label_lines):
        boxes = []
        for line in label_lines:
            class_index, cx, cy, w, h = list(map(float, line.split(' ')))
            class_index = int(class_index)
            boxes.append({
                'class_index': class_index,
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h,
                'area': w * h})
        return boxes

    @staticmethod
    def __sort_middle_last(big_last_boxes, small_last_boxes):
        middle_index = int(len(big_last_boxes) / 2)
        middle_last_boxes = []
        for i in range(middle_index):
            middle_last_boxes.append(big_last_boxes[i])
            middle_last_boxes.append(small_last_boxes[i])
        if len(big_last_boxes) % 2 == 1:
            middle_last_boxes.append(small_last_boxes[middle_index])
        return middle_last_boxes

    def __insert_batch_into_q(self):
        while True:
            fs, batch_x, batch_y1, batch_y2, batch_y3 = [], [], [], [], []
            for path in self.__get_next_batch_image_paths():
                fs.append(self.pool.submit(self.__load_img, path))
            for f in fs:
                cur_img_path, x = f.result()
                if x.shape[1] > self.input_shape[1] or x.shape[0] > self.input_shape[0]:
                    x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_AREA)
                else:
                    x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_LINEAR)
                x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
                batch_x.append(x)

                with open(f'{cur_img_path[:-4]}.txt', mode='rt') as file:
                    label_lines = file.readlines()

                boxes = self.__convert_to_boxes(label_lines)
                big_last_boxes = sorted(boxes, key=lambda __x: __x['area'], reverse=False)
                small_last_boxes = sorted(boxes, key=lambda __x: __x['area'], reverse=True)
                middle_last_boxes = self.__sort_middle_last(big_last_boxes, small_last_boxes)
                layer_mapping_boxes = [small_last_boxes, middle_last_boxes, big_last_boxes]

                y = []
                for i in range(len(self.output_shapes)):
                    y.append(np.zeros((self.output_shapes[i][1], self.output_shapes[i][2], self.output_shapes[i][3]), dtype=np.float32))

                for output_layer_index in range(len(self.output_shapes)):
                    output_rows = float(self.output_shapes[output_layer_index][1])
                    output_cols = float(self.output_shapes[output_layer_index][2])
                    for b in layer_mapping_boxes[output_layer_index]:
                        class_index, cx, cy, w, h = b['class_index'], b['cx'], b['cy'], b['w'], b['h']
                        grid_width_ratio = 1 / output_cols
                        grid_height_ratio = 1 / output_rows
                        center_row = int(cy * output_rows)
                        center_col = int(cx * output_cols)
                        y[output_layer_index][center_row][center_col][0] = 1.0
                        y[output_layer_index][center_row][center_col][1] = (cx - (center_col * grid_width_ratio)) / grid_width_ratio
                        y[output_layer_index][center_row][center_col][2] = (cy - (center_row * grid_height_ratio)) / grid_height_ratio
                        y[output_layer_index][center_row][center_col][3] = w
                        y[output_layer_index][center_row][center_col][4] = h
                        y[output_layer_index][center_row][center_col][int(class_index + 5)] = 1.0
                batch_y1.append(y[0])
                batch_y2.append(y[1])
                batch_y3.append(y[2])
            batch_x = np.asarray(batch_x).astype('float32')
            batch_y1 = np.asarray(batch_y1).astype('float32')
            batch_y2 = np.asarray(batch_y2).astype('float32')
            batch_y3 = np.asarray(batch_y3).astype('float32')
            sleep(0)
            # print(np.shape(batch_y1))
            # print(np.shape(batch_y2))
            # print(np.shape(batch_y3))
            # exit(1)
            self.batch_q.put((batch_x, [batch_y1, batch_y2, batch_y3]), block=True)

    def __get_next_batch_image_paths(self):
        start_index = self.batch_size * self.batch_index
        end_index = start_index + self.batch_size
        batch_image_paths = self.image_paths[start_index:end_index]
        self.batch_index += 1
        if self.batch_index == self.__len__():
            self.batch_index = 0
            np.random.shuffle(self.image_paths)
        return batch_image_paths

    def __load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)
        img = self.__random_adjust(img)  # so slow
        return path, img

    def __random_adjust(self, img):
        vs = [['hue', 0.1], ['saturation', 0.5], ['value', 0.5]]
        np.random.shuffle(vs)
        for v in vs:
            img = self.__adjust(img, v[1], v[0])
        return img

    @staticmethod
    def __adjust(img, adjust_val, adjust_type):
        range_min, range_max = 1.0 - adjust_val, 1.0 + adjust_val
        weight = np.random.uniform(range_min, range_max, 1)

        is_gray_img = False
        if len(img.shape) == 2:
            is_gray_img = True
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)

        if adjust_type == 'hue':
            h = np.asarray(h).astype('float32') * weight
            h = np.clip(h, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'saturation':
            s = np.asarray(s).astype('float32') * weight
            s = np.clip(s, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'value':
            v = np.asarray(v).astype('float32') * weight
            v = np.clip(v, 0.0, 255.0).astype('uint8')

        img = cv2.merge([h, s, v])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if is_gray_img:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
