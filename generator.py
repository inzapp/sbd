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
import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from logger import Logger
from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(
        self,
        teacher,
        image_paths,
        input_shape,
        output_shape,
        batch_size,
        num_workers,
        unknown_class_index,
        multi_classification_at_same_box,
        ignore_scale,
        virtual_anchor_iou_threshold,
        aug_scale,
        aug_mosaic,
        aug_h_flip,
        aug_v_flip,
        aug_brightness,
        aug_contrast,
        primary_device):
        assert 0.0 <= aug_brightness <= 1.0
        assert 0.0 <= aug_contrast <= 1.0
        assert 0.0 <= aug_scale <= 1.0
        assert 0.0 <= aug_mosaic <= 1.0
        self.debug = False
        self.teacher = teacher
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.input_height, self.input_width, self.input_channel = input_shape
        self.output_shapes = output_shape
        if type(self.output_shapes) == tuple:
            self.output_shapes = [self.output_shapes]
        self.num_classes = self.output_shapes[0][-1] - 5
        self.batch_size = batch_size
        self.unknown_class_index = unknown_class_index
        self.num_output_layers = len(self.output_shapes)
        self.multi_classification_at_same_box = multi_classification_at_same_box
        self.ignore_scale = ignore_scale
        self.virtual_anchor_iou_threshold = virtual_anchor_iou_threshold
        self.aug_scale = aug_scale
        self.aug_mosaic = aug_mosaic
        self.aug_h_flip = aug_h_flip
        self.aug_v_flip = aug_v_flip
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.primary_device = primary_device
        self.virtual_anchor_ws = []
        self.virtual_anchor_hs = []
        self.ws, self.hs = [], []
        self.img_index = 0
        self.pool = ThreadPoolExecutor(num_workers)
        np.random.shuffle(self.image_paths)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=aug_brightness, contrast_limit=aug_contrast),
            A.GaussianBlur(p=0.5, blur_limit=(5, 5))
        ])

    def label_path(self, image_path):
        return f'{image_path[:-4]}.txt'

    def is_label_exists(self, label_path):
        is_label_exists = False
        if os.path.exists(label_path) and os.path.isfile(label_path):
            is_label_exists = True
        return is_label_exists, label_path

    def load_label(self, label_path):
        labels = []
        label_exists = True
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            label_exists = False
        if label_exists:
            with open(label_path, 'rt') as f:
                lines = f.readlines()
                labels = [list(map(float, line.split())) for line in lines]
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
        return int(w * self.input_shape[1]) <= 3 or int(h * self.input_shape[0]) <= 3

    def check_label(self, image_paths, class_names, dataset_name):
        if self.teacher is not None:
            Logger.info(f'knowledge distillation training doesn\'t need label check, skip')
            return

        fs = []
        for path in image_paths:
            fs.append(self.pool.submit(self.load_label, self.label_path(path)))

        num_classes = self.num_classes
        if self.unknown_class_index > -1:
            num_classes += 1
            Logger.info(f'using unknown class with class index {self.unknown_class_index}')
        invalid_label_paths = set()
        not_found_label_paths = set()
        class_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        ignored_box_count = 0
        for f in tqdm(fs, desc=f'label check in {dataset_name} data'):
            labels, label_path, exists = f.result()
            if not exists:
                not_found_label_paths.add(label_path)
                continue
            for label in labels:
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

        if len(invalid_label_paths) > 0:
            print()
            for label_path in list(invalid_label_paths):
                print(label_path)
            Logger.error(f'{len(invalid_label_paths)} invalid label exists fix it')

        max_class_name_len = 0
        for name in class_names:
            max_class_name_len = max(max_class_name_len, len(name))
        if max_class_name_len == 0:
            max_class_name_len = 1

        Logger.info(f'class counts')
        for i in range(len(class_counts)):
            class_name = class_names[i]
            class_count = class_counts[i]
            Logger.info(f'{class_name:{max_class_name_len}s} : {class_count}')

        if dataset_name == 'train' and ignored_box_count > 0:
            Logger.warn(f'Too small size (under 3x3 pixel) {ignored_box_count} box will not be trained\n')
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
        if self.num_output_layers == 1 or self.virtual_anchor_iou_threshold == 0.0:
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

        if self.teacher is not None:
            self.virtual_anchor_ws = [0.5 for _ in range(self.num_output_layers)]
            self.virtual_anchor_hs = [0.5 for _ in range(self.num_output_layers)]
            Logger.info(f'knowledge distillation training doesn\'t need virtual anchor, skip')
            return

        if self.virtual_anchor_iou_threshold == 0.0:
            self.virtual_anchor_ws = [0.5 for _ in range(self.num_output_layers)]
            self.virtual_anchor_hs = [0.5 for _ in range(self.num_output_layers)]
            Logger.info(f'training with va_iou_threshold 0.0 doesn\'t need virtual anchor, skip')
            return

        self.ws = np.asarray(self.ws).reshape((len(self.ws), 1)).astype('float32')
        self.hs = np.asarray(self.hs).reshape((len(self.hs), 1)).astype('float32')

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
            Logger.warn(f'High IoU(>0.5) between virtual anchors may degrade mAP due to scale constraint. Consider using one output layer model instead\n')

        if print_avg_iou:
            fs = []
            for path in self.image_paths:
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

        if self.teacher is not None:
            Logger.info(f'knowledge distillation training doesn\'t need BPR, skip')
            return

        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, self.label_path(path)))

        y_true_obj_count = 0
        box_count_in_real_data = 0
        for f in tqdm(fs, desc='calculating BPR(Best Possible Recall)'):
            batch_y = [np.zeros(shape=(1,) + self.output_shapes[i][1:]) for i in range(self.num_output_layers)]
            batch_mask = [np.ones(shape=(1,) + self.output_shapes[i][1:]) for i in range(self.num_output_layers)]
            labels, _, _ = f.result()
            labeled_boxes = self.convert_to_boxes(labels)
            box_count_in_real_data += len(labeled_boxes)
            allocated_count = self.build_batch_tensor(labeled_boxes, batch_y, batch_mask, 0)
            y_true_obj_count += allocated_count

        avg_obj_count_per_image = box_count_in_real_data / float(len(self.image_paths))
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

    def random_scale(self, img, labels, scale_range):
        def overlay(img, overlay_img, start_x, start_y, channels):
            overlay_img_height, overlay_img_width = overlay_img.shape[:2]
            y_slice = slice(start_y, start_y + overlay_img_height)
            x_slice = slice(start_x, start_x + overlay_img_width)
            if channels == 1:
                img[y_slice, x_slice] = overlay_img[:overlay_img_height, :overlay_img_width]
            else:
                img[y_slice, x_slice, :] = overlay_img[:overlay_img_height, :overlay_img_width, :]
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
                black = np.zeros(shape=(self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
            else:
                black = np.zeros(shape=(self.input_shape[0], self.input_shape[1], channels), dtype=np.uint8)

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

    def random_flip(self, img, labels, aug_h_flip, aug_v_flip):
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

    def random_mosaic(self, datas):
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

    def augment(self, img, labels, first_call):
        if self.aug_brightness > 0.0 or self.aug_contrast > 0.0:
            img = self.transform(image=img)['image']

        if (self.aug_h_flip or self.aug_v_flip) and np.random.uniform() < 0.5:
            img, labels = self.random_flip(img, labels, self.aug_h_flip, self.aug_v_flip)

        if self.aug_scale > 0.0 and np.random.uniform() < 0.5:
            img, labels = self.random_scale(img, labels, self.aug_scale)

        if first_call:
            if self.aug_mosaic > 0.0 and np.random.uniform() < self.aug_mosaic:
                mosaic_data = self.load_batch_data(size=3, first_call=False)
                mosaic_data.append({'img': img, 'labels': labels})
                img, labels = self.random_mosaic(mosaic_data)
                if self.aug_scale > 0.0 and np.random.uniform() < 0.5:
                    img, labels = self.random_scale(img, labels, self.aug_scale)
        return img, labels

    def convert_to_boxes(self, labels):
        def get_same_box_index(labeled_boxes, cx, cy, w, h):
            if self.multi_classification_at_same_box:
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

    def build_batch_tensor(self, labeled_boxes, y, mask, batch_index, img=None):
        allocated_count = 0
        for b in labeled_boxes:
            class_indexes, cx, cy, w, h = b['class_indexes'], b['cx'], b['cy'], b['w'], b['h']
            if self.is_too_small_box(w, h):
                continue

            best_iou_indexes = self.get_iou_with_virtual_anchors([cx, cy, w, h])
            is_box_allocated = False
            for i, virtual_anchor_iou in best_iou_indexes:
                if is_box_allocated and virtual_anchor_iou < self.virtual_anchor_iou_threshold:
                    break
                output_rows = float(self.output_shapes[i][1])
                output_cols = float(self.output_shapes[i][2])
                rr, cc = np.meshgrid(np.arange(output_rows), np.arange(output_cols), indexing='ij')
                rr = np.asarray(rr).astype('float32') / (np.max(rr) + 1)
                cc = np.asarray(cc).astype('float32') / (np.max(cc) + 1)
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
                    center_only=y[i][batch_index][center_row][center_col][0] == 0.0)
                for grid in nearby_grids:
                    if grid['iou'] < 0.8:
                        break
                    offset_y = grid['offset_y']
                    offset_x = grid['offset_x']
                    cx_grid = grid['cx_grid']
                    cy_grid = grid['cy_grid']
                    offset_center_row = center_row + offset_y
                    offset_center_col = center_col + offset_x
                    if y[i][batch_index][offset_center_row][offset_center_col][0] == 0.0:
                        if 0.0 < self.ignore_scale <= 1.0:
                            half_scale = max(self.ignore_scale * 0.5, 1e-5)
                            object_heatmap = 1.0 - np.clip((np.abs(rr - center_row_f) / (h * half_scale)) ** 2 + (np.abs(cc - center_col_f) / (w * half_scale)) ** 2, 0.0, 1.0) ** 0.5
                            object_mask = np.where(object_heatmap == 0.0, 1.0, 0.0)

                            # confidence_channel = y[i][batch_index][:, :, 0]
                            # confidence_indices = np.where(object_heatmap > confidence_channel)
                            # confidence_channel[confidence_indices] = object_heatmap[confidence_indices]

                            confidence_mask_channel = mask[i][batch_index][:, :, 0]
                            confidence_mask_indices = np.where(object_mask == 0.0)
                            confidence_mask_channel[confidence_mask_indices] = object_mask[confidence_mask_indices]
                        y[i][batch_index][offset_center_row][offset_center_col][0] = 1.0
                        y[i][batch_index][offset_center_row][offset_center_col][1] = cx_grid
                        y[i][batch_index][offset_center_row][offset_center_col][2] = cy_grid
                        y[i][batch_index][offset_center_row][offset_center_col][3] = w
                        y[i][batch_index][offset_center_row][offset_center_col][4] = h
                        for class_index in class_indexes:
                            if class_index != self.unknown_class_index:
                                y[i][batch_index][center_row][center_col][class_index+5] = 1.0
                        is_box_allocated = True
                        allocated_count += 1
                        break
                mask[i][batch_index][:, :, 0][np.where(y[i][batch_index][:, :, 0] == 1.0)] = 1.0

        if self.debug:
            if self.input_shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(f'img.shape : {img.shape}')
            cv2.imshow('img', img)
            img_boxed = np.array(img)
            for bb in labeled_boxes:
                x1, y1, x2, y2 = self.cxcywh2x1y1x2y2(bb['cx'], bb['cy'], bb['w'], bb['h'])
                x1 = int(x1 * self.input_shape[1])
                y1 = int(y1 * self.input_shape[0])
                x2 = min(int(x2 * self.input_shape[1]), self.input_shape[1]-1)
                y2 = min(int(y2 * self.input_shape[0]), self.input_shape[0]-1)
                img_boxed = cv2.rectangle(img_boxed, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imshow('boxed', img_boxed)
            for i in range(self.num_output_layers):
                print(f'\n[layer_index, batch_index] : [{i}, {batch_index}]')
                confidence_channel = y[i][batch_index, :, :, 0]
                print(f'confidence_channel[{i}][{batch_index}].shape : {confidence_channel.shape}')
                mask_channel = mask[i][batch_index, :, :, 0]
                print(f'mask_channel[{i}][{batch_index}].shape : {mask_channel.shape}')
                for class_index in range(self.num_classes):
                    class_channel = y[i][batch_index, :, :, 5+class_index]
                    cv2.imshow(f'class_{class_index}[{i}]', cv2.resize(class_channel, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST))
                confidence_channel_img = cv2.resize(confidence_channel, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(f'confidence[{i}]', confidence_channel_img)
                cv2.imshow(f'mask[{i}]', cv2.resize(mask_channel, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST))

                if self.num_output_layers == 1:
                    alpha = 0.4
                    heatmap = np.clip(confidence_channel_img * 255.0, 0.0, 255.0).astype('uint8')
                    heatmap = heatmap.reshape(heatmap.shape + (1,))
                    heatmap = np.concatenate([heatmap, heatmap, heatmap], axis=-1)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    print(f'heatmap.shape : {heatmap.shape}')
                    if self.input_shape[-1] == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    blended_img = cv2.addWeighted(img, alpha, heatmap, 1.0 - alpha, 0)
                    cv2.imshow(f'heatmap[{i}]', blended_img)
                print(f'allocated_count : {allocated_count}\n')
            key = cv2.waitKey(0)
            if key == 27:
                exit(0)
        return allocated_count

    def load_image(self, path, with_bgr=False):
        bgr = None
        if with_bgr:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            bgr = img.copy()
            if self.input_shape[-1] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            color_mode = cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), color_mode)
        return img, bgr, path

    def preprocess(self, img, batch_axis=False):
        if self.input_shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        x = np.asarray(img).astype('float32') / 255.0
        if len(x.shape) == 2:
            x = x.reshape(x.shape + (1,))
        if batch_axis:
            x = x.reshape((1,) + x.shape)
        return x

    def load_batch_data(self, size, first_call):
        batch_data, fs = [], []
        for _ in range(size):
            fs.append(self.pool.submit(self.load_image, self.get_next_image_path()))
        for i in range(len(fs)):
            img, _, path = fs[i].result()
            img = self.resize(img, (self.input_width, self.input_height))
            labels, label_path, label_exists = self.load_label(self.label_path(path))
            if not label_exists:
                Logger.warn(f'label not found : {label_path}')
                continue
            img, labels = self.augment(img, labels, first_call=first_call)
            batch_data.append({'img': img, 'labels': labels})
        return batch_data
            
    def load(self):
        fs = []
        batch_x = np.zeros(shape=(self.batch_size,) + self.input_shape, dtype=np.float32)
        batch_tx, batch_y, batch_mask = None, None, None
        if self.teacher is None:
            batch_y = [np.zeros(shape=(self.batch_size,) + self.output_shapes[i][1:], dtype=np.float32) for i in range(self.num_output_layers)]
            batch_mask = [np.ones(shape=(self.batch_size,) + self.output_shapes[i][1:], dtype=np.float32) for i in range(self.num_output_layers)]
        else:
            if self.input_shape != self.teacher.input_shape[1:]:
                batch_tx = np.zeros(shape=(self.batch_size,) + self.teacher.input_shape[1:], dtype=np.float32)
        batch_data = self.load_batch_data(size=self.batch_size, first_call=True)
        for i in range(len(batch_data)):
            img = batch_data[i]['img']
            labels = batch_data[i]['labels']
            x = self.preprocess(img)
            batch_x[i] = x
            if self.teacher is None:
                labeled_boxes = self.convert_to_boxes(labels)
                self.build_batch_tensor(labeled_boxes, batch_y, batch_mask, i, img if self.debug else None)
            else:
                tx = None
                if self.input_shape != self.teacher.input_shape[1:]:
                    tw = self.teacher.input_shape[1:][1]
                    th = self.teacher.input_shape[1:][0]
                    tx = self.preprocess(self.resize(img, (tw, th)))
                    batch_tx[i] = tx
        if self.teacher is not None:
            from sbd import SBD
            batch_y = SBD.graph_forward(self.teacher, batch_x if batch_tx is None else batch_tx, self.primary_device)
            batch_mask = [1.0 for _ in range(self.num_output_layers)]
        if self.num_output_layers == 1:
            if self.teacher is None:
                batch_y = batch_y[0]
            batch_mask = batch_mask[0]
        return batch_x, batch_y, batch_mask

    def get_next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

