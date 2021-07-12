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
        np.random.shuffle(self.random_indexes)

    def __getitem__(self, index):
        batch_x = []
        batch_y1 = []
        batch_y2 = []
        batch_y3 = []
        start_index = index * self.batch_size
        fs = []
        for i in range(start_index, start_index + self.batch_size):
            fs.append(self.pool.submit(self.__load_img, self.image_paths[self.random_indexes[i]]))
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

            y = []
            for i in range(len(self.output_shapes)):
                y.append(np.zeros((self.output_shapes[i][1], self.output_shapes[i][2], self.output_shapes[i][3]), dtype=np.float32))
            for label_line in label_lines:
                class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
                if w > 0.3 or h > 0.3:
                    output_layer_index = 2
                elif w > 0.1 or h > 0.1:
                    output_layer_index = 1
                else:
                    output_layer_index = 0

                grid_width_ratio = 1 / float(self.output_shapes[output_layer_index][2])
                grid_height_ratio = 1 / float(self.output_shapes[output_layer_index][1])
                center_row = int(cy * self.output_shapes[output_layer_index][1])
                center_col = int(cx * self.output_shapes[output_layer_index][2])
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
        return batch_x, [batch_y1, batch_y2, batch_y3]

    def __load_img(self, path):
        return path, cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)

    def __len__(self):
        """
        Number of total iteration.
        """
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        """
        Mix the image paths at the end of each epoch.
        """
        np.random.shuffle(self.random_indexes)
