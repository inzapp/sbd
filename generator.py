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
import albumentations as A
from tqdm import tqdm
from util import ModelUtil


class YoloDataGenerator:
    def __init__(self, image_paths, input_shape, output_shape, batch_size, multi_classification_at_same_box):
        """
        :param input_shape:
            (height, width, channel) format of model input size
            If the channel is 1, train with a gray image, otherwise train with a color image.

        :param output_shape:
            Output shape extracted from built model.

        :param batch_size:
            Batch size of training.
        """
        self.generator_flow = GeneratorFlow(image_paths, input_shape, output_shape, batch_size, multi_classification_at_same_box)

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
    def __init__(self, image_paths, input_shape, output_shapes, batch_size, multi_classification_at_same_box):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.input_width, self.input_height, self.input_channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        self.output_shapes = output_shapes
        if type(self.output_shapes) == tuple:
            self.output_shapes = [self.output_shapes]
        if tf.keras.backend.image_data_format() == 'channels_first':
            self.num_classes = self.output_shapes[0][1] - 5
        else:
            self.num_classes = self.output_shapes[0][-1] - 5
        self.batch_size = batch_size
        self.num_output_layers = len(self.output_shapes)
        self.multi_classification_at_same_box = multi_classification_at_same_box
        self.virtual_anchor_ws = []
        self.virtual_anchor_hs = []
        self.batch_index = 0
        self.pool = ThreadPoolExecutor(8)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            A.GaussianBlur(p=0.5, blur_limit=(7, 7))
        ])

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
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, f'{path[:-4]}.txt'))
        invalid_label_paths = set()
        for f in tqdm(fs):
            lines, label_path = f.result()
            for line in lines:
                class_index, cx, cy, w, h = list(map(float, line.split()))
                if self.is_invalid_label(label_path, [class_index, cx, cy, w, h], self.num_classes):
                    invalid_label_paths.add(label_path)
        if len(invalid_label_paths) > 0:
            print()
            for label_path in list(invalid_label_paths):
                print(label_path)
            print(f'\n{len(invalid_label_paths)} invalid label exists fix it')
            exit(0)
        print('invalid label not found')

    def get_best_iou_layer_indexes(self, box):
        cx, cy, w, h = box
        x1 = cx - (w * 0.5)
        y1 = cy - (h * 0.5)
        x2 = cx + (w * 0.5)
        y2 = cy + (h * 0.5)
        labeled_box = [x1, y1, x2, y2]
        best_iou_layer_indexes = []
        for i in range(len(self.virtual_anchor_ws)):
            w = self.virtual_anchor_ws[i]
            h = self.virtual_anchor_hs[i]
            x1 = cx - (w * 0.5)
            y1 = cy - (h * 0.5)
            x2 = cx + (w * 0.5)
            y2 = cy + (h * 0.5)
            virtual_anchor_box = [x1, y1, x2, y2]
            iou = ModelUtil.iou(labeled_box, virtual_anchor_box)
            best_iou_layer_indexes.append([i, iou])
        return sorted(best_iou_layer_indexes, key=lambda x: x[1], reverse=True)

    def get_avg_iou_with_virtual_anchor(self, labeled_boxes):
        best_iou_sum = 0.0
        for box in labeled_boxes:
            best_iou_layer_indexes = self.get_best_iou_layer_indexes(box)
            best_iou = best_iou_layer_indexes[0][1]
            best_iou_sum += best_iou
        avg_iou = best_iou_sum / float(len(labeled_boxes))
        return avg_iou

    def calculate_virtual_anchor(self):
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, f'{path[:-4]}.txt'))

        labeled_boxes, ws, hs = [], [], []
        for f in tqdm(fs):
            lines, label_path = f.result()
            for line in lines:
                class_index, cx, cy, w, h = list(map(float, line.split()))
                labeled_boxes.append([cx, cy, w, h])
                ws.append(w)
                hs.append(h)

        ws = np.asarray(ws).reshape((1, len(ws))).astype('float32')
        hs = np.asarray(hs).reshape((1, len(hs))).astype('float32')

        num_cluster = self.num_output_layers
        criteria = (cv2.TERM_CRITERIA_EPS, -1, 1e-4)
        width_compactness, _, clustered_ws = cv2.kmeans(ws, num_cluster, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        width_compactness /= float(len(self.image_paths))
        print(f'width compactness  : {width_compactness:.7f}')
        height_compactness, _, clustered_hs = cv2.kmeans(hs, num_cluster, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        height_compactness /= float(len(self.image_paths))
        print(f'height compactness : {height_compactness:.7f}')

        self.virtual_anchor_ws = sorted(np.asarray(clustered_ws).reshape(-1))
        self.virtual_anchor_hs = sorted(np.asarray(clustered_hs).reshape(-1))

        print(f'virtual anchor : ', end='')
        for i in range(num_cluster):
            anchor_w = self.virtual_anchor_ws[i]
            anchor_h = self.virtual_anchor_hs[i]
            if i == 0:
                print(f'{anchor_w:.4f}, {anchor_h:.4f}', end='')
            else:
                print(f', {anchor_w:.4f}, {anchor_h:.4f}', end='')
        print()

        avg_iou_with_virtual_anchor = self.get_avg_iou_with_virtual_anchor(labeled_boxes)
        print(f'average IoU with virtual anchor : {avg_iou_with_virtual_anchor:.4f}')

    def calculate_best_possible_recall(self):
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, f'{path[:-4]}.txt'))

        y_true_obj_count = 0
        box_count_in_real_data = 0
        for f in tqdm(fs):
            label_lines, _ = f.result()
            labeled_boxes = self.convert_to_boxes(label_lines)
            box_count_in_real_data += len(labeled_boxes)
            _, allocated_count = self.build_batch_tensor(labeled_boxes)
            y_true_obj_count += allocated_count

        avg_obj_count_per_image = box_count_in_real_data / float(len(self.image_paths))
        y_true_obj_count = int(y_true_obj_count)
        not_trained_obj_count = box_count_in_real_data - (box_count_in_real_data if y_true_obj_count > box_count_in_real_data else y_true_obj_count)
        trained_obj_rate = y_true_obj_count / box_count_in_real_data * 100.0
        not_trained_obj_rate = not_trained_obj_count / box_count_in_real_data * 100.0
        best_possible_recall = y_true_obj_count / float(box_count_in_real_data)
        if best_possible_recall > 1.0:
            best_possible_recall = 1.0
        print(f'\naverage obj count per image : {avg_obj_count_per_image:.4f}\n')
        print(f'ground truth obj count : {box_count_in_real_data}')
        print(f'train tensor obj count : {y_true_obj_count} ({trained_obj_rate:.2f}%)')
        print(f'not trained  obj count : {not_trained_obj_count} ({not_trained_obj_rate:.2f}%)')
        print(f'best possible recall   : {best_possible_recall:.4f}')

    def convert_to_boxes(self, label_lines):
        def get_same_box_index(labeled_boxes, cx, cy, w, h):
            if self.multi_classification_at_same_box:
                box_str = f'{cx}_{cy}_{w}_{h}'
                for i in range(len(labeled_boxes)):
                    box_cx, box_cy, box_w, box_h = labeled_boxes[i]['cx'], labeled_boxes[i]['cy'], labeled_boxes[i]['w'], labeled_boxes[i]['h']
                    cur_box_str = f'{box_cx}_{box_cy}_{box_w}_{box_h}'
                    if cur_box_str == box_str:
                        return i
            return -1

        labeled_boxes = []
        for line in label_lines:
            class_index, cx, cy, w, h = list(map(float, line.split(' ')))
            class_index = int(class_index)
            same_box_index = get_same_box_index(labeled_boxes, cx, cy, w, h)
            if same_box_index == -1:
                labeled_boxes.append({
                    'class_indexes': [class_index],
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h,
                    'area': w * h})
            elif not class_index in labeled_boxes[same_box_index]['class_indexes']:
                labeled_boxes[same_box_index]['class_indexes'].append(class_index)
        return labeled_boxes

    def get_nearby_grids(self, confidence_channel, rows, cols, row, col, cx_grid, cy_grid, cx_raw, cy_raw, w, h):
        nearby_cells = []
        positions = [[-1, -1, 'lt'], [-1, 0, 't'], [-1, 1, 'rt'], [0, -1, 'l'], [0, 0, 'c'], [0, 1, 'r'], [1, -1, 'lb'], [1, 0, 'b'], [1, 1, 'rb']]
        for offset_y, offset_x, name in positions:
            if (0 <= row + offset_y < rows) and (0 <= col + offset_x < cols):
                if name == 'lt':
                    cx_nearby_grid = 1.0
                    cy_nearby_grid = 1.0
                elif name == 't':
                    cx_nearby_grid = cx_grid
                    cy_nearby_grid = 1.0
                elif name == 'rt':
                    cx_nearby_grid = 0.0
                    cy_nearby_grid = 1.0
                elif name == 'l':
                    cx_nearby_grid = 1.0
                    cy_nearby_grid = cy_grid
                elif name == 'c':
                    cx_nearby_grid = cx_grid
                    cy_nearby_grid = cy_grid
                elif name == 'r':
                    cx_nearby_grid = 0.0
                    cy_nearby_grid = cy_grid
                elif name == 'lb':
                    cx_nearby_grid = 1.0
                    cy_nearby_grid = 0.0
                elif name == 'b':
                    cx_nearby_grid = cx_grid
                    cy_nearby_grid = 0.0
                elif name == 'rb':
                    cx_nearby_grid = 0.0
                    cy_nearby_grid = 0.0

                box_origin = [
                    cx_raw - (w * 0.5),
                    cy_raw - (h * 0.5),
                    cx_raw + (w * 0.5),
                    cy_raw + (h * 0.5)]

                cx_nearby_raw = (float(col + offset_x) + cx_nearby_grid) / float(cols)
                cy_nearby_raw = (float(row + offset_y) + cy_nearby_grid) / float(rows)
                box_nearby = [
                    cx_nearby_raw - (w * 0.5),
                    cy_nearby_raw - (h * 0.5),
                    cx_nearby_raw + (w * 0.5),
                    cy_nearby_raw + (h * 0.5)]

                iou = ModelUtil.iou(box_origin, box_nearby)
                nearby_cells.append({
                    'offset_y': offset_y,
                    'offset_x': offset_x,
                    'cx_grid': cx_nearby_grid,
                    'cy_grid': cy_nearby_grid,
                    'iou': iou})
        return sorted(nearby_cells, key=lambda x: x['iou'], reverse=True)

    def build_batch_tensor(self, labeled_boxes):
        y = []
        for i in range(self.num_output_layers):
            y.append(np.zeros(shape=tuple(self.output_shapes[i][1:]), dtype=np.float32))

        allocated_count = 0
        image_data_format = tf.keras.backend.image_data_format()
        for b in labeled_boxes:
            class_indexes, cx, cy, w, h = b['class_indexes'], b['cx'], b['cy'], b['w'], b['h']
            best_iou_layer_indexes = self.get_best_iou_layer_indexes([cx, cy, w, h])
            is_box_allocated = False
            for i, layer_iou in best_iou_layer_indexes:
                if is_box_allocated and layer_iou < 0.6:
                    break
                output_rows = float(self.output_shapes[i][1])
                output_cols = float(self.output_shapes[i][2])
                center_row = int(cy * output_rows)
                center_col = int(cx * output_cols)
                cx_grid_scale = (cx - float(center_col) / output_cols) / (1.0 / output_cols)
                cy_grid_scale = (cy - float(center_row) / output_rows) / (1.0 / output_rows)
                if image_data_format == 'channels_first':
                    nearby_grids = self.get_nearby_grids(
                        confidence_channel=y[i][0, :, :],
                        rows=output_rows,
                        cols=output_cols,
                        row=center_row,
                        col=center_col,
                        cx_grid=cx_grid_scale,
                        cy_grid=cy_grid_scale,
                        cx_raw=cx,
                        cy_raw=cy,
                        w=w,
                        h=h)
                    for grid in nearby_grids:
                        if grid['iou'] < 0.8:
                            break
                        offset_y = grid['offset_y']
                        offset_x = grid['offset_x']
                        cx_grid = grid['cx_grid']
                        cy_grid = grid['cy_grid']
                        if y[i][0][center_row+offset_y][center_col+offset_x] == 1.0:
                            y[i][0][center_row+offset_y][center_col+offset_x] = 1.0
                            y[i][1][center_row+offset_y][center_col+offset_x] = cx_grid
                            y[i][2][center_row+offset_y][center_col+offset_x] = cy_grid
                            y[i][3][center_row+offset_y][center_col+offset_x] = w
                            y[i][4][center_row+offset_y][center_col+offset_x] = h
                            for class_index in class_indexes:
                                y[i][class_index+5][center_row][center_col] = 1.0
                            is_box_allocated = True
                            allocated_count += 1
                            break
                else:
                    nearby_grids = self.get_nearby_grids(
                        confidence_channel=y[i][:, :, 0],
                        rows=output_rows,
                        cols=output_cols,
                        row=center_row,
                        col=center_col,
                        cx_grid=cx_grid_scale,
                        cy_grid=cy_grid_scale,
                        cx_raw=cx,
                        cy_raw=cy,
                        w=w,
                        h=h)
                    for grid in nearby_grids:
                        if grid['iou'] < 0.8:
                            break
                        offset_y = grid['offset_y']
                        offset_x = grid['offset_x']
                        cx_grid = grid['cx_grid']
                        cy_grid = grid['cy_grid']
                        if y[i][center_row+offset_y][center_col+offset_x][0] == 0.0:
                            y[i][center_row+offset_y][center_col+offset_x][0] = 1.0
                            y[i][center_row+offset_y][center_col+offset_x][1] = cx_grid
                            y[i][center_row+offset_y][center_col+offset_x][2] = cy_grid
                            y[i][center_row+offset_y][center_col+offset_x][3] = w
                            y[i][center_row+offset_y][center_col+offset_x][4] = h
                            for class_index in class_indexes:
                                y[i][center_row][center_col][class_index+5] = 1.0
                            is_box_allocated = True
                            allocated_count += 1
                            break
        return y, allocated_count
            
    def __getitem__(self, index):
        fs, batch_x, batch_y = [], [], []
        for i in range(self.num_output_layers):
            batch_y.append([])
        for path in self.get_next_batch_image_paths():
            fs.append(self.pool.submit(ModelUtil.load_img, path, self.input_channel))
        for f in fs:
            img, _, cur_img_path = f.result()
            img = self.transform(image=img)['image']
            img = ModelUtil.resize(img, (self.input_width, self.input_height))
            x = ModelUtil.preprocess(img)
            batch_x.append(x)

            with open(f'{cur_img_path[:-4]}.txt', mode='rt') as file:
                label_lines = file.readlines()
            labeled_boxes = self.convert_to_boxes(label_lines)
            np.random.shuffle(labeled_boxes)

            y, _ = self.build_batch_tensor(labeled_boxes)
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

