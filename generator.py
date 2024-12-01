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
import os
import sys
import cv2
import signal
import threading
import numpy as np
import albumentations as A

from glob import glob
from tqdm import tqdm
from time import sleep
from logger import Logger
from collections import deque
from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self, cfg, output_shape, class_names, unknown_class_index, training=False, debug=False):
        assert 0.0 <= cfg.aug_noise <= 1.0
        assert 0.0 <= cfg.aug_scale <= 1.0
        assert 0.0 <= cfg.aug_mosaic <= 1.0
        assert 0.0 <= cfg.aug_contrast <= 1.0
        assert 0.0 <= cfg.aug_brightness <= 1.0
        assert 0.0 <= cfg.aug_snowstorm <= 1.0
        self.cfg = cfg
        self.training = training
        self.debug = debug
        self.data_paths = self.get_data_paths()
        self.output_shapes = output_shape
        if type(self.output_shapes) == tuple:
            self.output_shapes = [self.output_shapes]
        self.num_classes = self.output_shapes[0][-1] - 5
        self.class_names = class_names
        self.unknown_class_index = unknown_class_index
        self.num_output_layers = len(self.output_shapes)

        self.data_index = 0
        self.virtual_anchor_ws = []
        self.virtual_anchor_hs = []
        self.ws, self.hs = [], []
        self.lock = threading.Lock()
        self.q_thread = threading.Thread(target=self.load_xy_into_q)
        self.q_thread.daemon = True
        self.q = deque()
        self.q_thread_running = False
        self.q_thread_pause = False
        self.q_indices = list(range(self.cfg.max_q_size))
        self.pool = ThreadPoolExecutor(8)

        self.class_weights = None
        self.use_class_weights = self.training and self.cfg.cls_balance > 0.0
        if self.training:
            np.random.shuffle(self.data_paths)
        self.transform = A.Compose([
            A.ToGray(p=0.01),
            A.RandomBrightnessContrast(brightness_limit=self.cfg.aug_brightness, contrast_limit=0.0, p=0.5),
            A.Lambda(name='augment_noise', image=self.augment_noise, p=0.5),
            A.Lambda(name='augment_contrast', image=self.augment_contrast, p=0.5),
            A.Lambda(name='augment_snowstorm', image=self.augment_snowstorm, p=self.cfg.aug_snowstorm),
            A.GaussianBlur(p=0.5, blur_limit=(5, 5))
        ])

    def get_data_paths(self):
        if self.training:
            data_path = self.cfg.train_data_path
        else:
            data_path = self.cfg.validation_data_path

        if data_path.endswith('.txt'):
            with open(data_path, 'rt') as f:
                data_paths = f.readlines()
            for i in range(len(data_paths)):
                data_paths[i] = data_paths[i].replace('\n', '')
        else:
            data_paths = glob(f'{data_path}/**/*.jpg', recursive=True)
        return data_paths

    def label_path(self, data_path):
        return f'{data_path[:-4]}.txt'

    def is_label_exists(self, label_path):
        is_label_exists = False
        if os.path.exists(label_path) and os.path.isfile(label_path):
            is_label_exists = True
        return is_label_exists, label_path

    def remove_duplicate_labels(self, labels):
        unique_labels = set(tuple(label) for label in labels)
        return [list(label) for label in unique_labels]

    def load_label(self, label_path, remove_duplicate=True):
        labels = []
        label_exists = True
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            label_exists = False
        if label_exists:
            with open(label_path, 'rt') as f:
                lines = f.readlines()
            labels = [list(map(float, line.split())) for line in lines]
            if remove_duplicate:
                labels = self.remove_duplicate_labels(labels)
        return labels, label_path, label_exists

    def is_invalid_label(self, path, label, num_classes):
        class_index, cx, cy, w, h = label
        if class_index < 0 or class_index >= num_classes:
            Logger.warn(f'\ninvalid class index {int(class_index)} in num_classs {num_classes} : [{path}]')
            return True
        elif cx <= 0.0 or cx >= 1.0 or cy <= 0.0 or cy >= 1.0:
            Logger.warn(f'\ninvalid cx or cy. cx : {cx:.6f}, cy : {cy:.6f} : [{path}]')
            return True
        elif w <= 0.0 or w > 1.0 or h <= 0.0 or h > 1.0:
            Logger.warn(f'\ninvalid width or height. width : {w:.6f}, height : {h:.6f} : [{path}]')
            return True
        else:
            return False

    def is_too_small_box(self, w, h):
        return int(w * self.cfg.input_cols) <= 3 or int(h * self.cfg.input_rows) <= 3

    def calculate_class_weights(self, class_counts_param, gamma):
        class_counts = np.array(class_counts_param, dtype=np.float32)
        class_counts[class_counts == 0] = np.max(class_counts)
        weights = 1.0 / class_counts
        weights = weights ** gamma
        median_weight = np.median(weights)
        weights = weights / median_weight
        return weights

    def check_label(self):
        fs = []
        for path in self.data_paths:
            fs.append(self.pool.submit(self.load_label, self.label_path(path), remove_duplicate=False))

        num_classes = self.num_classes
        if self.unknown_class_index > -1:
            num_classes += 1
            Logger.info(f'using unknown class with class index {self.unknown_class_index}')

        invalid_label_paths = set()
        duplicate_label_paths = set()
        not_found_label_paths = set()
        class_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        ignored_box_count = 0

        dataset_name = 'train' if self.training else 'validation'
        for f in tqdm(fs, desc=f'label check in {dataset_name} data'):
            labels, label_path, exists = f.result()
            if not exists:
                not_found_label_paths.add(label_path)
                continue

            unique_labels = self.remove_duplicate_labels(labels)
            if len(unique_labels) < len(labels):
                duplicate_label_paths.add((label_path, len(labels) - len(unique_labels)))

            for label in unique_labels:
                class_index, cx, cy, w, h = label
                class_counts[int(class_index)] += 1
                if self.is_invalid_label(label_path, [class_index, cx, cy, w, h], num_classes):
                    invalid_label_paths.add(label_path)
                if self.is_too_small_box(w, h):
                    ignored_box_count += 1
                elif self.num_output_layers > 1:
                    self.ws.append(w)
                    self.hs.append(h)

        if len(not_found_label_paths) > 0:
            print()
            for label_path in list(not_found_label_paths):
                Logger.warn(f'label not found : {label_path}')
            Logger.error(f'{len(not_found_label_paths)} labels not found')

        if len(duplicate_label_paths) > 0:
            print()
            for label_path, duplicate_count in list(duplicate_label_paths):
                Logger.warn(f'{duplicate_count} duplicate labels removed : {label_path}')

        if len(invalid_label_paths) > 0:
            print()
            for label_path in list(invalid_label_paths):
                print(label_path)
            Logger.error(f'{len(invalid_label_paths)} invalid label exists fix it')

        max_class_name_len = 0
        for name in self.class_names:
            max_class_name_len = max(max_class_name_len, len(name))
        if max_class_name_len == 0:
            max_class_name_len = 1

        if self.use_class_weights:
            self.class_weights = self.calculate_class_weights(class_counts, self.cfg.cls_balance)

        class_count_txts = []
        if self.use_class_weights:
            class_count_txts.append(f'{dataset_name} data class count(class balance gamma {self.cfg.cls_balance})')
        else:
            class_count_txts.append(f'{dataset_name} data class count')

        for i in range(len(class_counts)):
            class_name = self.class_names[i]
            class_count = class_counts[i]
            if self.use_class_weights:
                class_count_txts.append(f'{class_name:{max_class_name_len}s} : {class_count} => {self.class_weights[i]:.2f}')
            else:
                class_count_txts.append(f'{class_name:{max_class_name_len}s} : {class_count}')
        Logger.info(class_count_txts)

        if dataset_name == 'train' and ignored_box_count > 0:
            Logger.warn(f'Too small size (under 3 pixel) {ignored_box_count} box will not be trained\n')
        else:
            print()

    def iou(self, a, b):
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

    def cxcywh2x1y1x2y2(self, cx, cy, w, h):
        x1 = cx - (w * 0.5)
        y1 = cy - (h * 0.5)
        x2 = cx + (w * 0.5)
        y2 = cy + (h * 0.5)
        return x1, y1, x2, y2

    def get_iou_with_virtual_anchors(self, box):
        if self.num_output_layers == 1 or self.cfg.va_iou_threshold == 0.0:
            return [[i, 1.0] for i in range(self.num_output_layers)]

        cx, cy, w, h = box
        x1, y1, x2, y2 = self.cxcywh2x1y1x2y2(cx, cy, w, h)
        labeled_box = np.clip(np.asarray([x1, y1, x2, y2]), 0.0, 1.0)
        iou_with_virtual_anchors = []
        for layer_index in range(self.num_output_layers):
            w = self.virtual_anchor_ws[layer_index]
            h = self.virtual_anchor_hs[layer_index]
            x1, y1, x2, y2 = self.cxcywh2x1y1x2y2(cx, cy, w, h)
            virtual_anchor_box = np.clip(np.asarray([x1, y1, x2, y2]), 0.0, 1.0)
            iou = self.iou(labeled_box, virtual_anchor_box)
            iou_with_virtual_anchors.append([layer_index, iou])
        return sorted(iou_with_virtual_anchors, key=lambda x: x[1], reverse=True)

    def calculate_virtual_anchor(self, print_avg_iou=False):
        if self.num_output_layers == 1:  # one layer model doesn't need virtual anchor
            self.virtual_anchor_ws = [0.5]
            self.virtual_anchor_hs = [0.5]
            Logger.info('skip calculating virtual anchor when output layer size is 1')
            return

        if self.cfg.va_iou_threshold == 0.0:
            self.virtual_anchor_ws = [0.5 for _ in range(self.num_output_layers)]
            self.virtual_anchor_hs = [0.5 for _ in range(self.num_output_layers)]
            Logger.info(f'training with va_iou_threshold 0.0 doesn\'t need virtual anchor, skip')
            return

        self.ws = np.asarray(self.ws).reshape((len(self.ws), 1)).astype(np.float32)
        self.hs = np.asarray(self.hs).reshape((len(self.hs), 1)).astype(np.float32)

        max_iterations = 100
        num_cluster = self.num_output_layers
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 1e-4)

        Logger.info('K-means clustering start')
        w_sse, _, clustered_ws = cv2.kmeans(self.ws, num_cluster, None, criteria, max_iterations, cv2.KMEANS_RANDOM_CENTERS)
        w_mse = w_sse / (float(len(self.ws)) + 1e-5)
        h_sse, _, clustered_hs = cv2.kmeans(self.hs, num_cluster, None, criteria, max_iterations, cv2.KMEANS_RANDOM_CENTERS)
        h_mse = h_sse / (float(len(self.hs)) + 1e-5)
        clustering_mse = (w_mse + h_mse) / 2.0
        Logger.info(f'clustered MSE(Mean Squared Error) : {clustering_mse:.4f}')

        self.virtual_anchor_ws = sorted(np.asarray(clustered_ws).reshape(-1), reverse=True)
        self.virtual_anchor_hs = sorted(np.asarray(clustered_hs).reshape(-1), reverse=True)
        del self.ws
        del self.hs

        Logger.info('virtual anchor : ', end='')
        for i in range(num_cluster):
            anchor_w = self.virtual_anchor_ws[i]
            anchor_h = self.virtual_anchor_hs[i]
            if i == 0:
                print(f'{anchor_w:.4f}, {anchor_h:.4f}', end='')
            else:
                print(f', {anchor_w:.4f}, {anchor_h:.4f}', end='')
        print('\n')

        iou_between_va_sum = 0.0
        Logger.info('IoU between virtual anchors')
        for i in range(num_cluster - 1):
            box_a = self.cxcywh2x1y1x2y2(0.5, 0.5, self.virtual_anchor_ws[i], self.virtual_anchor_hs[i])
            box_b = self.cxcywh2x1y1x2y2(0.5, 0.5, self.virtual_anchor_ws[i+1], self.virtual_anchor_hs[i+1])
            iou = self.iou(box_a, box_b)
            iou_between_va_sum += iou
            Logger.info(f'va[{i}], va[{i+1}] => {iou:.4f}')
        avg_iou_between_va = iou_between_va_sum / (num_cluster - 1)
        Logger.info(f'average IoU between virtual anchor : {avg_iou_between_va:.4f}\n')
        if avg_iou_between_va > 0.5:
            Logger.warn(f'high IoU(>0.5) between virtual anchors may degrade mAP due to scale constraint. consider using one output layer model instead\n')

        if print_avg_iou:
            fs = []
            for path in self.data_paths:
                fs.append(self.pool.submit(self.load_label, self.label_path(path)))
            labeled_boxes = []
            for f in tqdm(fs, desc='load box data for calculating avg IoU'):
                labels, label_path, _ = f.result()
                for label in labels:
                    class_index, cx, cy, w, h = label
                    if not self.is_too_small_box(w, h):
                        labeled_boxes.append([cx, cy, w, h])

            best_iou_sum = 0.0
            for box in tqdm(labeled_boxes, desc='average IoU with virtual anchors'):
                iou_with_virtual_anchors = self.get_iou_with_virtual_anchors(box)
                best_iou = iou_with_virtual_anchors[0][1]
                best_iou_sum += best_iou
            avg_iou_with_virtual_anchor = best_iou_sum / (float(len(labeled_boxes)) + 1e-5)
            Logger.info(f'average IoU : {avg_iou_with_virtual_anchor:.4f}\n')

    def calculate_best_possible_recall(self):
        if self.debug:
            return

        fs = []
        for path in self.data_paths:
            fs.append(self.pool.submit(self.load_label, self.label_path(path)))

        y_true_obj_count = 0
        box_count_in_real_data = 0
        for f in tqdm(fs, desc='calculating BPR(Best Possible Recall)'):
            batch_y = [np.zeros(shape=self.output_shapes[i][1:]) for i in range(self.num_output_layers)]
            batch_extra = [np.ones(shape=self.output_shapes[i][1:]) for i in range(self.num_output_layers)]
            labels, _, _ = f.result()
            labeled_boxes = self.convert_to_boxes(labels)
            box_count_in_real_data += len(labeled_boxes)
            allocated_count = self.build_gt_tensor(labeled_boxes, batch_y, batch_extra, 0)
            y_true_obj_count += allocated_count

        avg_obj_count_per_image = box_count_in_real_data / float(len(self.data_paths))
        y_true_obj_count = int(y_true_obj_count)
        not_trained_obj_count = box_count_in_real_data - (box_count_in_real_data if y_true_obj_count > box_count_in_real_data else y_true_obj_count)
        trained_obj_rate = y_true_obj_count / box_count_in_real_data * 100.0
        not_trained_obj_rate = not_trained_obj_count / box_count_in_real_data * 100.0
        best_possible_recall = y_true_obj_count / float(box_count_in_real_data)
        if best_possible_recall > 1.0:
            best_possible_recall = 1.0
        Logger.info(f'ground truth obj count : {box_count_in_real_data}')
        Logger.info(f'train tensor obj count : {y_true_obj_count} ({trained_obj_rate:.2f}%)')
        Logger.info(f'not trained  obj count : {not_trained_obj_count} ({not_trained_obj_rate:.2f}%)')
        Logger.info(f'best possible recall   : {best_possible_recall:.4f}')
        Logger.info(f'average obj count per image : {avg_obj_count_per_image:.4f}\n')

    def resize(self, img, size):
        img_h, img_w = img.shape[:2]
        if img_h > size[0] or img_w > size[1]:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        return img

    def augment_noise(self, img, **kwargs):
        if self.cfg.aug_noise > 0.0:
            img = np.array(img).astype(np.float32)
            noise_power = np.random.uniform() * (self.cfg.aug_noise * 255.0)
            img_h, img_w = img.shape[:2]
            img = img.reshape((img_h, img_w, -1))
            img += np.random.uniform(-noise_power, noise_power, size=(img_h, img_w, self.cfg.input_channels))
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img

    def augment_contrast(self, img, **kwargs):
        if self.cfg.aug_contrast > 0.0:
            power = np.random.uniform() * self.cfg.aug_contrast
            img_f = np.asarray(img).astype(np.float32)
            contrast_offset = (127.5 - img_f) * power
            if np.random.uniform() < 0.5:
                img_f += contrast_offset
            else:
                img_f -= contrast_offset
            img = np.clip(img_f, 0.0, 255.0).astype(np.uint8)
        return img

    def augment_snowstorm(self, img, **kwargs):
        img_h, img_w = img.shape[:2]
        num_snowflakes_range = (50, 200)
        snowflake_length_range = (10, max(min(img_w, img_h) // 3, 10))
        curvature_range = (1, 15)
        direction_angle_range = (-50, 50)
        color_range = (180, 230)
        thickness_range = (1, max(min(img_w, img_h) // 192, 2))

        num_snowflakes = np.random.randint(num_snowflakes_range[0], num_snowflakes_range[1])
        for _ in range(num_snowflakes):
            snowflake_length = np.random.randint(snowflake_length_range[0], snowflake_length_range[1])
            curvature = np.random.randint(curvature_range[0], curvature_range[1])
            x = np.random.randint(0, img_w - snowflake_length)
            y = np.random.randint(0, img_h - snowflake_length)

            start_x = x
            end_x = x + snowflake_length
            start_y = y + snowflake_length // 2

            num_points = snowflake_length * 2
            t = np.linspace(0, 1, num_points)
            curve = curvature * np.sin(np.pi * t)

            direction_angle = np.random.randint(direction_angle_range[0], direction_angle_range[1])
            snowflake_points = np.column_stack((np.linspace(start_x, end_x, num_points), start_y + curve))
            rotation_matrix = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), direction_angle, 1)
            snowflake_points = cv2.transform(np.array([snowflake_points]), rotation_matrix)[0]

            color_val = np.random.randint(color_range[0], color_range[1])
            thickness = np.random.randint(thickness_range[0], thickness_range[1])
            color = (color_val, color_val, color_val)
            cv2.polylines(img, [snowflake_points.astype(np.int32)], isClosed=False, color=color, thickness=thickness)
        return img

    def augment_scale(self, img, labels, scale_range):
        def overlay(img, overlay_img, start_x, start_y, channels):
            overlay_img_h, overlay_img_w = overlay_img.shape[:2]
            y_slice = slice(start_y, start_y + overlay_img_h)
            x_slice = slice(start_x, start_x + overlay_img_w)
            if channels == 1:
                img[y_slice, x_slice] = overlay_img[:overlay_img_h, :overlay_img_w]
            else:
                img[y_slice, x_slice, :] = overlay_img[:overlay_img_h, :overlay_img_w, :]
            return img

        scale_range = max(scale_range, 0.01)
        max_scale = 1.0
        min_scale = 1.0 - scale_range
        scale = np.random.uniform() * (max_scale - min_scale) + min_scale
        img_h, img_w = img.shape[:2]
        channels = 1
        if len(img.shape) == 3 and img.shape[-1] == 3:
            channels = 3

        scaled_h, scaled_w = int(img_h * scale), int(img_w * scale)
        start_x = np.random.randint(img_w - scaled_w)
        start_y = np.random.randint(img_h - scaled_h)

        roi_x1 = start_x / float(img_w)
        roi_y1 = start_y / float(img_h)
        roi_x2 = (start_x + scaled_w) / float(img_w)
        roi_y2 = (start_y + scaled_h) / float(img_h)
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        new_labels = []
        if np.random.uniform() < 0.5:  # downscale
            reduced_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if channels == 1:
                black = np.zeros(shape=(self.cfg.input_rows, self.cfg.input_cols), dtype=np.uint8)
            else:
                black = np.zeros(shape=(self.cfg.input_rows, self.cfg.input_cols, channels), dtype=np.uint8)

            scaled_img = overlay(black, reduced_img, start_x, start_y, channels)
            for label in labels:
                class_index, cx, cy, w, h = label
                class_index = int(class_index)
                cx *= roi_w
                cy *= roi_h
                cx += roi_x1
                cy += roi_y1
                w *= roi_w
                h *= roi_h
                cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
                new_labels.append([class_index, cx, cy, w, h])
        else:  # upscale
            scaled_img = cv2.resize(img[start_y:start_y+scaled_h, start_x:start_x+scaled_w], (img_w, img_h), cv2.INTER_LINEAR)
            for label in labels:
                class_index, cx, cy, w, h = label
                class_index = int(class_index)
                x1, y1, x2, y2 = self.cxcywh2x1y1x2y2(cx, cy, w, h)

                x1 = np.clip(x1, roi_x1, roi_x2)
                y1 = np.clip(y1, roi_y1, roi_y2)
                x2 = np.clip(x2, roi_x1, roi_x2)
                y2 = np.clip(y2, roi_y1, roi_y2)

                x1 = (x1 - roi_x1) / roi_w
                y1 = (y1 - roi_y1) / roi_h
                x2 = (x2 - roi_x1) / roi_w
                y2 = (y2 - roi_y1) / roi_h
                x1, y1, x2, y2 = np.clip(np.array([x1, y1, x2, y2]), 0.0, 1.0)

                w = x2 - x1
                h = y2 - y1
                if w > 0.0 and h > 0.0:
                    cx = x1 + (w * 0.5)
                    cy = y1 + (h * 0.5)
                    cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
                    new_labels.append([class_index, cx, cy, w, h])
        return scaled_img, new_labels

    def augment_flip(self, img, labels, aug_h_flip, aug_v_flip):
        method = ''
        if aug_h_flip and aug_v_flip:
            method = 'a'
        elif aug_h_flip:
            method = 'h'
        elif aug_v_flip:
            method = 'v'

        aug_method = np.random.choice(['h', 'v', 'a']) if method == 'a' else method
        if aug_method == 'h':
            img = cv2.flip(img, 1)
        elif aug_method == 'v':
            img = cv2.flip(img, 0)
        elif aug_method == 'a':
            img = cv2.flip(img, -1)

        new_labels = []
        for label in labels:
            class_index, cx, cy, w, h = label
            class_index = int(class_index)
            if aug_method in ['h', 'a']:
                cx = 1.0 - cx
            if aug_method in ['v', 'a']:
                cy = 1.0 - cy
            cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
            new_labels.append([class_index, cx, cy, w, h])
        return img, new_labels

    def augment_mosaic(self, datas):
        np.random.shuffle(datas)
        img_0 = cv2.resize(datas[0]['img'], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img_1 = cv2.resize(datas[1]['img'], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img_2 = cv2.resize(datas[2]['img'], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img_3 = cv2.resize(datas[3]['img'], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img = np.concatenate([np.concatenate([img_0, img_1], axis=1), np.concatenate([img_2, img_3], axis=1)], axis=0)

        new_labels = []
        for i in range(len(datas)):
            labels = datas[i]['labels']
            for label in labels:
                class_index, cx, cy, w, h = label
                cx *= 0.5
                cy *= 0.5
                w *= 0.5
                h *= 0.5
                if i == 0:  # left top
                    pass
                elif i == 1:  # right top
                    cx += 0.5
                elif i == 2:  # left bottom
                    cy += 0.5
                elif i == 3:  # right bottom
                    cx += 0.5
                    cy += 0.5
                else:
                    Logger.warn(f'invalid mosaic index : {i}')
                cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
                new_labels.append([class_index, cx, cy, w, h])
        return img, new_labels

    def augment_mixup(self, datas, alpha=0.5):
        np.random.shuffle(datas)
        img_0 = datas[0]['img']
        img_1 = datas[1]['img']
        img = cv2.addWeighted(img_0, alpha, img_1, 1 - alpha, 0)

        new_labels = []
        for i in range(len(datas)):
            labels = datas[i]['labels']
            for label in labels:
                class_index, cx, cy, w, h = label
                cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
                new_labels.append([class_index, cx, cy, w, h])
        return img, new_labels

    def augment(self, img, labels, multi_image_augmentation):
        if self.cfg.aug_brightness > 0.0 or self.cfg.aug_contrast > 0.0:
            img = self.transform(image=img)['image']

        if (self.cfg.aug_h_flip or self.cfg.aug_v_flip) and np.random.uniform() < 0.5:
            img, labels = self.augment_flip(img, labels, self.cfg.aug_h_flip, self.cfg.aug_v_flip)

        if self.cfg.aug_scale > 0.0 and np.random.uniform() < 0.5:
            img, labels = self.augment_scale(img, labels, self.cfg.aug_scale)

        if multi_image_augmentation:
            if self.cfg.aug_mosaic > 0.0 and np.random.uniform() < self.cfg.aug_mosaic:
                mosaic_data = self.load_image_with_label(size=3, multi_image_augmentation=False)
                mosaic_data.append({'img': img, 'labels': labels})
                img, labels = self.augment_mosaic(mosaic_data)
                if self.cfg.aug_scale > 0.0 and np.random.uniform() < 0.5:
                    img, labels = self.augment_scale(img, labels, self.cfg.aug_scale)

            if self.cfg.aug_mixup > 0.0 and np.random.uniform() < self.cfg.aug_mixup:
                mixup_data = self.load_image_with_label(size=1, multi_image_augmentation=False)
                mixup_data.append({'img': img, 'labels': labels})
                img, labels = self.augment_mixup(mixup_data)
        return img, labels

    def convert_to_boxes(self, labels):
        def get_same_box_index(labeled_boxes, cx, cy, w, h):
            if self.cfg.multi_classification_at_same_box:
                box_str = f'{cx:.6f}_{cy:.6f}_{w:.6f}_{h:.6f}'
                for i in range(len(labeled_boxes)):
                    box_cx, box_cy, box_w, box_h = labeled_boxes[i]['cx'], labeled_boxes[i]['cy'], labeled_boxes[i]['w'], labeled_boxes[i]['h']
                    cur_box_str = f'{box_cx:.6f}_{box_cy:.6f}_{box_w:.6f}_{box_h:.6f}'
                    if cur_box_str == box_str:
                        return i
            return -1

        labeled_boxes = []
        for label in labels:
            class_index, cx, cy, w, h = label
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
        return sorted(labeled_boxes, key=lambda x: x['area'], reverse=True)

    def get_nearby_grids(self, rows, cols, row, col, cx_grid, cy_grid, cx_raw, cy_raw, w, h, center_only):
        positions = None
        if center_only:
            positions = [[0, 0, 'c']]
        else:
            positions = [[-1, -1, 'lt'], [-1, 0, 't'], [-1, 1, 'rt'], [0, -1, 'l'], [0, 1, 'r'], [1, -1, 'lb'], [1, 0, 'b'], [1, 1, 'rb']]
        nearby_cells = []
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

                if name == 'c':
                    iou = 1.0
                else:
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
                    box_nearby = np.clip(np.array(box_nearby), 0.0, 1.0)
                    iou = self.iou(box_origin, box_nearby) - 1e-4  # subtract small value for give center grid to first priority
                nearby_cells.append({
                    'offset_y': offset_y,
                    'offset_x': offset_x,
                    'cx_grid': cx_nearby_grid,
                    'cy_grid': cy_nearby_grid,
                    'iou': iou})
        return sorted(nearby_cells, key=lambda x: x['iou'], reverse=True)

    def blend_heatmap(self, img, objectness, alpha=0.4):
        img = np.asarray(img)
        objectness = np.asarray(objectness)

        img_h, img_w = img.shape[:2]
        objectness_h, objectness_w = objectness.shape[:2]

        objectness_img = np.clip(objectness * 255.0, 0.0, 255.0).astype(np.uint8).reshape((objectness_h, objectness_w, 1))
        objectness_img = cv2.cvtColor(objectness_img, cv2.COLOR_GRAY2BGR)
        objectness_img = cv2.resize(objectness_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        heatmap = cv2.applyColorMap(objectness_img, cv2.COLORMAP_JET)

        img = img.reshape((img_h, img_w, -1))
        if img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        blended_img = cv2.addWeighted(img, alpha, heatmap, 1.0 - alpha, 0)
        return blended_img

    def build_gt_tensor(self, labeled_boxes, y, extra, img=None):
        allocated_count = 0
        for b in labeled_boxes:
            class_indexes, cx, cy, w, h = b['class_indexes'], b['cx'], b['cy'], b['w'], b['h']
            if self.is_too_small_box(w, h):
                continue

            best_iou_indexes = self.get_iou_with_virtual_anchors([cx, cy, w, h])
            is_box_allocated = False
            for i, virtual_anchor_iou in best_iou_indexes:
                if is_box_allocated and virtual_anchor_iou < self.cfg.va_iou_threshold:
                    break
                output_rows = float(self.output_shapes[i][1])
                output_cols = float(self.output_shapes[i][2])
                rr, cc = np.meshgrid(np.arange(output_rows), np.arange(output_cols), indexing='ij')
                rr = np.asarray(rr).astype(np.float32) / (np.max(rr) + 1)
                cc = np.asarray(cc).astype(np.float32) / (np.max(cc) + 1)
                center_row = int(cy * output_rows)
                center_col = int(cx * output_cols)
                center_row_f = int(cy * output_rows) / output_rows
                center_col_f = int(cx * output_cols) / output_cols
                cx_grid_scale = (cx - float(center_col) / output_cols) / (1.0 / output_cols)
                cy_grid_scale = (cy - float(center_row) / output_rows) / (1.0 / output_rows)
                nearby_grids = self.get_nearby_grids(
                    rows=output_rows,
                    cols=output_cols,
                    row=center_row,
                    col=center_col,
                    cx_grid=cx_grid_scale,
                    cy_grid=cy_grid_scale,
                    cx_raw=cx,
                    cy_raw=cy,
                    w=w,
                    h=h,
                    center_only=y[i][center_row][center_col][0] == 0.0)
                for grid in nearby_grids:
                    if grid['iou'] < 0.8:
                        break
                    offset_y = grid['offset_y']
                    offset_x = grid['offset_x']
                    cx_grid = grid['cx_grid']
                    cy_grid = grid['cy_grid']
                    offset_center_row = center_row + offset_y
                    offset_center_col = center_col + offset_x
                    if y[i][offset_center_row][offset_center_col][0] == 0.0:
                        if self.cfg.obj_target == 'binary' and 0.0 < self.cfg.heatmap_scale <= 1.0:
                            half_scale = max(self.cfg.heatmap_scale * 0.5, 1e-5)
                            object_heatmap = 1.0 - np.clip((np.abs(rr - center_row_f) / (h * half_scale)) ** 2 + (np.abs(cc - center_col_f) / (w * half_scale)) ** 2, 0.0, 1.0) ** 0.5
                            object_mask = np.where(object_heatmap == 0.0, 1.0, 0.0)

                            confidence_channel = y[i][:, :, 0]
                            confidence_indices = np.where(object_heatmap > confidence_channel)
                            confidence_channel[confidence_indices] = object_heatmap[confidence_indices]
                            # for class_index in class_indexes:
                            #     if class_index != self.unknown_class_index:
                            #         class_channel = y[i][:, :, class_index+5]
                            #         class_indices = np.where(object_heatmap > class_channel)
                            #         class_channel[class_indices] = object_heatmap[class_indices]

                            # confidence_mask_channel = extra[i][:, :, 0]
                            # confidence_mask_indices = np.where(object_mask == 0.0)
                            # confidence_mask_channel[confidence_mask_indices] = object_mask[confidence_mask_indices]
                        y[i][offset_center_row][offset_center_col][0] = 1.0
                        y[i][offset_center_row][offset_center_col][1] = cx_grid
                        y[i][offset_center_row][offset_center_col][2] = cy_grid
                        y[i][offset_center_row][offset_center_col][3] = w
                        y[i][offset_center_row][offset_center_col][4] = h
                        for class_index in class_indexes:
                            if class_index != self.unknown_class_index:
                                y[i][center_row][center_col][class_index+5] = 1.0
                        is_box_allocated = True
                        allocated_count += 1
                        break
                extra[i][:, :, 0][np.where(y[i][:, :, 0] == 1.0)] = 1.0

        if self.use_class_weights:
            for i in range(len(y)):
                for class_index in range(self.num_classes):
                    extra[i][:, :, class_index+5] = self.class_weights[class_index]

        if self.debug:
            print(f'img.shape : {img.shape}')
            cv2.imshow('img', img)
            img_boxed = np.array(img)
            for bb in labeled_boxes:
                x1, y1, x2, y2 = self.cxcywh2x1y1x2y2(bb['cx'], bb['cy'], bb['w'], bb['h'])
                x1 = int(x1 * self.cfg.input_cols)
                y1 = int(y1 * self.cfg.input_rows)
                x2 = min(int(x2 * self.cfg.input_cols), self.cfg.input_cols-1)
                y2 = min(int(y2 * self.cfg.input_rows), self.cfg.input_rows-1)
                img_boxed = cv2.rectangle(img_boxed, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow('boxed', img_boxed)
            for i in range(self.num_output_layers):
                print(f'\nlayer_index : {i}')
                objectness = y[i][:, :, 0]
                print(f'objectness[{i}]shape : {objectness.shape}')
                mask_channel = extra[i][:, :, 0]
                print(f'mask_channel[{i}].shape : {mask_channel.shape}')
                # for class_index in range(self.num_classes):
                #     class_channel = y[i][:, :, 5+class_index]
                #     cv2.imshow(f'class_{class_index}[{i}]', cv2.resize(class_channel, (self.cfg.input_cols, self.cfg.input_rows), interpolation=cv2.INTER_NEAREST))
                objectness_img = cv2.resize(objectness, (self.cfg.input_cols, self.cfg.input_rows), interpolation=cv2.INTER_NEAREST)
                # cv2.imshow(f'confidence[{i}]', objectness_img)
                # cv2.imshow(f'extra[{i}]', cv2.resize(mask_channel, (self.cfg.input_cols, self.cfg.input_rows), interpolation=cv2.INTER_NEAREST))

                if self.num_output_layers == 1:
                    blended_img = self.blend_heatmap(img, objectness)
                    cv2.imshow(f'heatmap[{i}]', blended_img)

                print(f'allocated_count : {allocated_count}\n')
            key = cv2.waitKey(0)
            if key == 27:
                self.exit()
        return allocated_count

    def load_image(self, path, gray=False):
        color_mode = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), color_mode)
        return img, path

    def preprocess(self, img, batch_axis=False):
        if self.cfg.input_channels == 1 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.cfg.input_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        x = np.asarray(img).astype(np.float32) / 255.0
        if len(x.shape) == 2:
            x = x.reshape(x.shape + (1,))
        if batch_axis:
            x = x.reshape((1,) + x.shape)
        return x

    def next_data_path(self):
        path = self.data_paths[self.data_index]
        self.data_index += 1
        if self.data_index == len(self.data_paths):
            self.data_index = 0
            np.random.shuffle(self.data_paths)
        return path

    def load_image_with_label(self, size, multi_image_augmentation):
        data, fs = [], []
        for _ in range(size):
            fs.append(self.pool.submit(self.load_image, self.next_data_path(), gray=self.cfg.input_channels == 1))
        for i in range(len(fs)):
            img, path = fs[i].result()
            img = self.resize(img, (self.cfg.input_cols, self.cfg.input_rows))
            labels, label_path, label_exists = self.load_label(self.label_path(path))
            if not label_exists:
                Logger.warn(f'label not found : {label_path}')
                continue
            if self.training:
                img, labels = self.augment(img, labels, multi_image_augmentation=multi_image_augmentation)
            data.append({'img': img, 'labels': labels})
        return data

    def signal_handler(self, sig, frame):
        print()
        Logger.info(f'{signal.Signals(sig).name} signal detected, please wait until the end of the thread')
        self.stop()
        Logger.info(f'exit successfully')
        sys.exit(0)

    def start(self):
        if self.debug:
            return
        self.q_thread_running = True
        self.q_thread.start()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        while True:
            sleep(1.0)
            percentage = (len(self.q) / self.cfg.max_q_size) * 100.0
            Logger.info(f'prefetching training data... {percentage:.1f}%')
            with self.lock:
                if len(self.q) >= self.cfg.max_q_size:
                    print()
                    break

    def stop(self):
        if self.q_thread_running:
            self.q_thread_running = False
            while self.q_thread.is_alive():
                sleep(0.1)

    def pause(self):
        if self.q_thread_running:
            self.q_thread_pause = True

    def resume(self):
        if self.q_thread_running:
            self.q_thread_pause = False

    def exit(self):
        self.signal_handler(signal.SIGINT, None)

    def load_xy(self):
        y = [np.zeros(shape=self.output_shapes[i][1:], dtype=np.float32) for i in range(self.num_output_layers)]
        extra = [np.ones(shape=self.output_shapes[i][1:], dtype=np.float32) for i in range(self.num_output_layers)]
        img_with_label = self.load_image_with_label(size=1, multi_image_augmentation=True)
        img = img_with_label[0]['img']
        labels = img_with_label[0]['labels']
        x = self.preprocess(img)
        labeled_boxes = self.convert_to_boxes(labels)
        self.build_gt_tensor(labeled_boxes, y, extra, img if self.debug else None)
        return x, y, extra
            
    def load_xy_into_q(self):
        while self.q_thread_running:
            if self.q_thread_pause:
                sleep(1.0)
            else:
                x, y, extra = self.load_xy()
                with self.lock:
                    if len(self.q) == self.cfg.max_q_size:
                        self.q.popleft()
                    self.q.append((x, y, extra))

    def load(self):
        batch_x = []
        if self.num_output_layers == 1:
            batch_y, batch_e = [], []
        else:
            batch_y = [[] for _ in range(self.num_output_layers)]
            batch_e = [[] for _ in range(self.num_output_layers)]
        for i in np.random.choice(self.q_indices, self.cfg.batch_size, replace=False):
            with self.lock:
                if self.debug:
                    x, y, m = self.load_xy()
                else:
                    x, y, m = self.q[i]
                batch_x.append(np.array(x))
                if self.num_output_layers == 1:
                    batch_y.append(np.array(y))
                    batch_e.append(np.array(m))
                else:
                    for j in range(self.num_output_layers):
                        batch_y[j].append(np.array(y[j]))
                        batch_e[j].append(np.array(m[j]))
        batch_x = np.asarray(batch_x).reshape((self.cfg.batch_size, self.cfg.input_rows, self.cfg.input_cols, self.cfg.input_channels)).astype(np.float32)
        if self.num_output_layers == 1:
            batch_y = np.asarray(batch_y).reshape((self.cfg.batch_size,) + self.output_shapes[0][1:]).astype(np.float32)
            batch_e = np.asarray(batch_e).reshape((self.cfg.batch_size,) + self.output_shapes[0][1:]).astype(np.float32)
        else:
            for i in range(self.num_output_layers):
                batch_y[i] = np.asarray(batch_y[i]).reshape((self.cfg.batch_size,) + self.output_shapes[i][1:]).astype(np.float32)
                batch_e[i] = np.asarray(batch_e[i]).reshape((self.cfg.batch_size,) + self.output_shapes[i][1:]).astype(np.float32)
        return batch_x, batch_y, batch_e

