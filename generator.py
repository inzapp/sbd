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
        if type(self.output_shapes) == tuple:
            self.output_shapes = [self.output_shapes]
        self.batch_size = batch_size
        self.num_output_layers = len(self.output_shapes)
        self.virtual_anchor_ws = []
        self.virtual_anchor_hs = []
        self.label_obj_count = 0  # obj count in real label txt
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

    def get_iou_with_index(self, box, virtual_anchor_ws, virtual_anchor_hs):
        cx, cy, w, h = box
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        true_box = [x1, y1, x2, y2]
        res = []
        for i in range(self.num_output_layers):
            x1 = cx - virtual_anchor_ws[i] * 0.5
            y1 = cy - virtual_anchor_hs[i] * 0.5
            x2 = cx + virtual_anchor_ws[i] * 0.5
            y2 = cy + virtual_anchor_hs[i] * 0.5
            virtual_anchor_box = [x1, y1, x2, y2]
            iou = self.iou(true_box, virtual_anchor_box)
            res.append({'index': i, 'iou': iou})
        return sorted(res, key=lambda x: x['iou'], reverse=True)

    def get_layer_indexes_threshold(self, box, virtual_anchor_ws, virtual_anchor_hs, threshold=0.6):
        res = self.get_iou_with_index(box, virtual_anchor_ws, virtual_anchor_hs)
        indexes = [res[0]['index']]
        for i in range(1, len(res)):
            if res[i]['iou'] >= threshold:
                indexes.append(res[i]['index'])
        return indexes

    def get_best_iou_with_index(self, box, virtual_anchor_ws, virtual_anchor_hs):
        res = self.get_iou_with_index(box, virtual_anchor_ws, virtual_anchor_hs)
        return res[0]['iou'], res[0]['index']

    def iou_with_clustered_wh(self, boxes, virtual_anchor_ws, virtual_anchor_hs):
        iou_sum = 0.0
        for true_box in boxes:
            best_iou, _ = self.get_best_iou_with_index(true_box, virtual_anchor_ws, virtual_anchor_hs)
            iou_sum += best_iou
        return iou_sum / float(len(boxes))

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

    def cluster_wh(self):
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, f'{path[:-4]}.txt'))

        num_classes = self.output_shapes[0][-1] - 5
        invalid_label_paths = set()
        boxes, ws, hs = [], [], []
        for f in tqdm(fs):
            lines, label_path = f.result()
            for line in lines:
                class_index, cx, cy, w, h = list(map(float, line.split()))
                if self.is_invalid_label(label_path, [class_index, cx, cy, w, h], num_classes):
                    invalid_label_paths.add(label_path)
                boxes.append([cx, cy, w, h])
                ws.append(w)
                hs.append(h)
                self.label_obj_count += 1

        if len(invalid_label_paths) > 0:
            for label_path in list(invalid_label_paths):
                print(label_path)
            print('\ninvalid label exists fix it')
            exit(0)

        ws = np.asarray(ws).reshape((1, len(ws))).astype('float32')
        hs = np.asarray(hs).reshape((1, len(hs))).astype('float32')

        num_cluster = self.num_output_layers
        criteria = (cv2.TERM_CRITERIA_EPS, -1, 1e-4)
        width_compactness, _, ws = cv2.kmeans(ws, num_cluster, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        width_compactness /= float(len(self.image_paths))
        print(f'width compactness  : {width_compactness:.7f}')
        height_compactness, _, hs = cv2.kmeans(hs, num_cluster, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        height_compactness /= float(len(self.image_paths))
        print(f'height compactness : {height_compactness:.7f}')

        self.virtual_anchor_ws = sorted(np.asarray(ws).reshape(-1))
        self.virtual_anchor_hs = sorted(np.asarray(hs).reshape(-1))

        print(f'clustered  widths : ', end='')
        for i in range(num_cluster):
            print(f'{self.virtual_anchor_ws[i]:.4f} ', end='')
        print()
        print(f'clustered heights : ', end='')
        for i in range(num_cluster):
            print(f'{self.virtual_anchor_hs[i]:.4f} ', end='')
        print()
        print(f'avg IoU : {self.iou_with_clustered_wh(boxes, self.virtual_anchor_ws, self.virtual_anchor_hs):.4f}')

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

        y_true_obj_count = 0  # obj count in train tensor(y_true)
        for batch_x, batch_y in tqdm(self):
            if self.num_output_layers == 1:
                batch_y = [batch_y]
            max_area = 0
            max_area_index = -1
            for i in range(self.num_output_layers):
                cur_area = np.sum(np.ones_like(batch_y[i][:, :, :, 0], dtype=np.float32))
                if cur_area > max_area:
                    max_area = cur_area
                    max_area_index = i
            y_true_obj_count += np.sum(batch_y[max_area_index][:, :, :, 0])
        y_true_obj_count = int(y_true_obj_count)
        not_trained_obj_count = box_count_in_real_data - y_true_obj_count
        not_trained_obj_rate = not_trained_obj_count / box_count_in_real_data * 100.0
        print(f'ground truth obj count : {box_count_in_real_data}')
        print(f'train tensor obj count : {y_true_obj_count}')
        print(f'not trained  obj count : {not_trained_obj_count} ({not_trained_obj_rate:.2f}%)')

    @staticmethod
    def convert_to_boxes(label_lines):
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
            else:
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
                fs.append(self.pool.submit(self.load_img, path, self.input_shape[-1]))
            for f in fs:
                x, _, cur_img_path = f.result()
                x = self.random_blur(x)
                if x.shape[1] > self.input_shape[1] or x.shape[0] > self.input_shape[0]:
                    x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_AREA)
                else:
                    x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_LINEAR)
                x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
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
                    
                for output_layer_index in range(self.num_output_layers):
                    output_rows = float(self.output_shapes[output_layer_index][1])
                    output_cols = float(self.output_shapes[output_layer_index][2])
                    for b in layer_mapping_boxes[output_layer_index]:
                        class_indexes, cx, cy, w, h = b['class_indexes'], b['cx'], b['cy'], b['w'], b['h']
                        center_row = int(cy * output_rows)
                        center_col = int(cx * output_cols)
                        y[output_layer_index][center_row][center_col][0] = 1.0
                        y[output_layer_index][center_row][center_col][1] = (cx - float(center_col) / output_cols) / (1.0 / output_cols)
                        y[output_layer_index][center_row][center_col][2] = (cy - float(center_row) / output_rows) / (1.0 / output_rows)
                        y[output_layer_index][center_row][center_col][3] = w
                        y[output_layer_index][center_row][center_col][4] = h
                        for class_index in class_indexes:
                            y[output_layer_index][center_row][center_col][class_index+5] = 1.0

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

    @staticmethod
    def load_img(path, channel):
        color_mode = cv2.IMREAD_COLOR if channel == 3 else cv2.IMREAD_GRAYSCALE
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), color_mode)
        raw_bgr = img
        if color_mode == cv2.IMREAD_COLOR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        return img, raw_bgr, path

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

