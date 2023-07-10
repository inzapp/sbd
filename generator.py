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
from util import Util
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
        aug_h_flip,
        aug_v_flip,
        aug_brightness,
        aug_contrast,
        primary_device):
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
        self.aug_h_flip = aug_h_flip
        self.aug_v_flip = aug_v_flip
        self.primary_device = primary_device
        self.virtual_anchor_ws = []
        self.virtual_anchor_hs = []
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
        lines = []
        label_exists = True
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            label_exists = False
        if label_exists:
            with open(label_path, 'rt') as f:
                lines = f.readlines()
        return lines, label_path, label_exists

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

    def check_label(self, image_paths, class_names, dataset_name):
        if self.teacher is not None:
            print(f'knowledge distillation training doesn\'t need label check, skip')
            return

        fs = []
        for path in image_paths:
            fs.append(self.pool.submit(self.load_label, self.label_path(path)))

        num_classes = self.num_classes
        if self.unknown_class_index > -1:
            num_classes += 1
            print(f'using unknown class with class index {self.unknown_class_index}')
        invalid_label_paths = set()
        not_found_label_paths = set()
        class_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        for f in tqdm(fs, desc=f'label check in {dataset_name} data'):
            lines, label_path, exists = f.result()
            if not exists:
                not_found_label_paths.add(label_path)
            if len(not_found_label_paths) == 0:
                for line in lines:
                    class_index, cx, cy, w, h = list(map(float, line.split()))
                    class_counts[int(class_index)] += 1
                    if self.is_invalid_label(label_path, [class_index, cx, cy, w, h], num_classes):
                        invalid_label_paths.add(label_path)

        if len(not_found_label_paths) > 0:
            print()
            for label_path in list(not_found_label_paths):
                print(f'label not found : {label_path}')
            Util.print_error_exit(f'{len(not_found_label_paths)} labels not found')

        if len(invalid_label_paths) > 0:
            print()
            for label_path in list(invalid_label_paths):
                print(label_path)
            Util.print_error_exit(f'{len(invalid_label_paths)} invalid label exists fix it')

        max_class_name_len = 0
        for name in class_names:
            max_class_name_len = max(max_class_name_len, len(name))
        if max_class_name_len == 0:
            max_class_name_len = 1

        print(f'class counts')
        for i in range(len(class_counts)):
            class_name = class_names[i]
            class_count = class_counts[i]
            print(f'{class_name:{max_class_name_len}s} : {class_count}')
        print()

    def get_iou_with_virtual_anchors(self, box):
        if self.num_output_layers == 1 or self.virtual_anchor_iou_threshold == 0.0:
            return [[i, 1.0] for i in range(self.num_output_layers)]

        cx, cy, w, h = box
        x1 = cx - (w * 0.5)
        y1 = cy - (h * 0.5)
        x2 = cx + (w * 0.5)
        y2 = cy + (h * 0.5)
        labeled_box = np.clip(np.asarray([x1, y1, x2, y2]), 0.0, 1.0)
        iou_with_virtual_anchors = []
        for layer_index in range(self.num_output_layers):
            w = self.virtual_anchor_ws[layer_index]
            h = self.virtual_anchor_hs[layer_index]
            x1 = cx - (w * 0.5)
            y1 = cy - (h * 0.5)
            x2 = cx + (w * 0.5)
            y2 = cy + (h * 0.5)
            virtual_anchor_box = np.clip(np.asarray([x1, y1, x2, y2]), 0.0, 1.0)
            iou = Util.iou(labeled_box, virtual_anchor_box)
            iou_with_virtual_anchors.append([layer_index, iou])
        return sorted(iou_with_virtual_anchors, key=lambda x: x[1], reverse=True)

    def calculate_virtual_anchor(self, print_avg_iou=False):
        if self.num_output_layers == 1:  # one layer model doesn't need virtual anchor
            self.virtual_anchor_ws = [0.5]
            self.virtual_anchor_hs = [0.5]
            print('skip calculating virtual anchor when output layer size is 1')
            return

        if self.teacher is not None:
            self.virtual_anchor_ws = [0.5 for _ in range(self.num_output_layers)]
            self.virtual_anchor_hs = [0.5 for _ in range(self.num_output_layers)]
            print(f'knowledge distillation training doesn\'t need virtual anchor, skip')
            return

        if self.virtual_anchor_iou_threshold == 0.0:
            self.virtual_anchor_ws = [0.5 for _ in range(self.num_output_layers)]
            self.virtual_anchor_hs = [0.5 for _ in range(self.num_output_layers)]
            print(f'training with va_iou_threshold 0.0 doesn\'t need virtual anchor, skip')
            return

        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, self.label_path(path)))
        ignore_box_count = 0
        labeled_boxes, ws, hs = [], [], []
        for f in tqdm(fs, desc='load box size for calculating virtual anchor'):
            lines, label_path, _ = f.result()
            for line in lines:
                class_index, cx, cy, w, h = list(map(float, line.split()))
                is_w_valid = int(w * self.input_shape[1]) > 3
                if is_w_valid:
                    ws.append(w)
                is_h_valid = int(h * self.input_shape[0]) > 3
                if is_h_valid:
                    hs.append(h)
                if print_avg_iou:
                    if is_w_valid and is_h_valid:
                        labeled_boxes.append([cx, cy, w, h])
                    else:
                        ignore_box_count += 1
        if ignore_box_count > 0:
            print(f'[Warning] Too small size (under 3x3 pixel) {ignore_box_count} box will not be trained')

        ws = np.asarray(ws).reshape((len(ws), 1)).astype('float32')
        hs = np.asarray(hs).reshape((len(hs), 1)).astype('float32')

        max_iterations = 100
        num_cluster = self.num_output_layers
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 1e-4)

        print('K-means clustering start')
        w_sse, _, clustered_ws = cv2.kmeans(ws, num_cluster, None, criteria, max_iterations, cv2.KMEANS_RANDOM_CENTERS)
        w_mse = w_sse / (float(len(ws)) + 1e-5)
        h_sse, _, clustered_hs = cv2.kmeans(hs, num_cluster, None, criteria, max_iterations, cv2.KMEANS_RANDOM_CENTERS)
        h_mse = h_sse / (float(len(hs)) + 1e-5)
        clustering_mse = (w_mse + h_mse) / 2.0
        print(f'clustered MSE(Mean Squared Error) : {clustering_mse:.4f}')

        self.virtual_anchor_ws = sorted(np.asarray(clustered_ws).reshape(-1), reverse=True)
        self.virtual_anchor_hs = sorted(np.asarray(clustered_hs).reshape(-1), reverse=True)

        print(f'virtual anchor : ', end='')
        for i in range(num_cluster):
            anchor_w = self.virtual_anchor_ws[i]
            anchor_h = self.virtual_anchor_hs[i]
            if i == 0:
                print(f'{anchor_w:.4f}, {anchor_h:.4f}', end='')
            else:
                print(f', {anchor_w:.4f}, {anchor_h:.4f}', end='')
        print('\n')

        if print_avg_iou:
            best_iou_sum = 0.0
            for box in tqdm(labeled_boxes, desc='average IoU with virtual anchors'):
                iou_with_virtual_anchors = self.get_iou_with_virtual_anchors(box)
                best_iou = iou_with_virtual_anchors[0][1]
                best_iou_sum += best_iou
            avg_iou_with_virtual_anchor = best_iou_sum / (float(len(labeled_boxes)) + 1e-5)
            print(f'average IoU : {avg_iou_with_virtual_anchor:.4f}\n')

    def calculate_best_possible_recall(self):
        if self.debug:
            return

        if self.teacher is not None:
            print(f'knowledge distillation training doesn\'t need BPR, skip')
            return

        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, self.label_path(path)))

        y_true_obj_count = 0
        box_count_in_real_data = 0
        for f in tqdm(fs, desc='calculating BPR(Best Possible Recall)'):
            batch_y = [np.zeros(shape=(1,) + self.output_shapes[i][1:]) for i in range(self.num_output_layers)]
            batch_mask = [np.ones(shape=(1,) + self.output_shapes[i][1:]) for i in range(self.num_output_layers)]
            label_lines, _, _ = f.result()
            labeled_boxes = self.convert_to_boxes(label_lines)
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
        print(f'ground truth obj count : {box_count_in_real_data}')
        print(f'train tensor obj count : {y_true_obj_count} ({trained_obj_rate:.2f}%)')
        print(f'not trained  obj count : {not_trained_obj_count} ({not_trained_obj_rate:.2f}%)')
        print(f'best possible recall   : {best_possible_recall:.4f}')
        print(f'\naverage obj count per image : {avg_obj_count_per_image:.4f}\n')

    def random_scale(self, img, label_lines, min_scale):
        def overlay(img, overlay_img, start_x, start_y, channels):
            overlay_img_height, overlay_img_width = overlay_img.shape[:2]
            y_slice = slice(start_y, start_y + overlay_img_height)
            x_slice = slice(start_x, start_x + overlay_img_width)
            if channels == 1:
                img[y_slice, x_slice] = overlay_img[:overlay_img_height, :overlay_img_width]
            else:
                img[y_slice, x_slice, :] = overlay_img[:overlay_img_height, :overlay_img_width, :]
            return img

        max_scale = 1.0
        scale = np.random.uniform() * (max_scale - min_scale) + min_scale
        img_height, img_width = img.shape[:2]
        channels = 1
        if len(img.shape) == 3 and img.shape[-1] == 3:
            channels = 3

        reduced_height, reduced_width = int(img_height * scale), int(img_width * scale)
        reduced_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if channels == 1:
            black = np.zeros(shape=(self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        else:
            black = np.zeros(shape=(self.input_shape[0], self.input_shape[1], channels), dtype=np.uint8)

        start_x = np.random.randint(img_width - reduced_width)
        start_y = np.random.randint(img_height - reduced_height)
        scaled_img = overlay(black, reduced_img, start_x, start_y, channels)

        roi_x1 = start_x / float(img_width)
        roi_y1 = start_y / float(img_height)
        roi_x2 = (start_x + reduced_width) / float(img_width)
        roi_y2 = (start_y + reduced_height) / float(img_height)
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        scaled_label_lines = []
        for line in label_lines:
            class_index, cx, cy, w, h = list(map(float, line.split()))
            class_index = int(class_index)
            cx *= roi_w
            cy *= roi_h
            cx += roi_x1
            cy += roi_y1
            w *= roi_w
            h *= roi_h
            cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
            scaled_label_lines.append(f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        return scaled_img, scaled_label_lines

    def random_flip(self, img, label_lines, aug_h_flip, aug_v_flip):
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

        converted_label_lines = []
        for line in label_lines:
            class_index, cx, cy, w, h = list(map(float, line.split()))
            class_index = int(class_index)
            if aug_method in ['h', 'a']:
                cx = 1.0 - cx
            if aug_method in ['v', 'a']:
                cy = 1.0 - cy
            cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
            converted_label_lines.append(f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        return img, converted_label_lines

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
                    iou = Util.iou(box_origin, box_nearby) - 1e-4  # subtract small value for give center grid to first priority
                nearby_cells.append({
                    'offset_y': offset_y,
                    'offset_x': offset_x,
                    'cx_grid': cx_nearby_grid,
                    'cy_grid': cy_nearby_grid,
                    'iou': iou})
        return sorted(nearby_cells, key=lambda x: x['iou'], reverse=True)

    def build_batch_tensor(self, labeled_boxes, y, mask, batch_index):
        allocated_count = 0
        for b in labeled_boxes:
            class_indexes, cx, cy, w, h = b['class_indexes'], b['cx'], b['cy'], b['w'], b['h']
            w_not_valid = int(w * self.input_shape[1]) <= 3
            h_not_valid = int(h * self.input_shape[0]) <= 3
            if w_not_valid and h_not_valid:
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
                            object_mask[offset_center_row][offset_center_col] = 1.0
                            confidence_mask_channel = mask[i][batch_index][:, :, 0]
                            confidence_mask_indices = np.where(object_mask < confidence_mask_channel)
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
        if self.debug:
            confidence_channel = y[i][batch_index, :, :, 0]
            print(f'confidence_channel.shape : {confidence_channel.shape}')
            mask_channel = mask[i][batch_index, :, :, 0]
            print(f'mask_channel.shape : {mask_channel.shape}')
            for class_index in range(self.num_classes):
                class_channel = y[i][batch_index, :, :, 5+class_index]
                cv2.imshow(f'class_{class_index}', cv2.resize(class_channel, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST))
            cv2.imshow('confidence', cv2.resize(confidence_channel, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST))
            cv2.imshow('mask', cv2.resize(mask_channel, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST))
            print(f'allocated_count : {allocated_count}')
        return allocated_count
            
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
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(Util.load_img, self.get_next_image_path(), self.input_channel))
        for i in range(len(fs)):
            img, _, path = fs[i].result()
            if self.debug:
                cv2.imshow('img', cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.input_channel == 3 else img, (self.input_shape[1], self.input_shape[0])))
            img = Util.resize(img, (self.input_width, self.input_height))
            img = self.transform(image=img)['image']
            label_lines, label_path, label_exists = self.load_label(self.label_path(path))
            if not label_exists:
                print(f'label not found : {label_path}')
                continue
            if self.aug_scale < 1.0 and np.random.uniform() < 0.5:
                img, label_lines = self.random_scale(img, label_lines, self.aug_scale)
            if (self.aug_h_flip or self.aug_v_flip) and np.random.uniform() < 0.5:
                img, label_lines = self.random_flip(img, label_lines, self.aug_h_flip, self.aug_v_flip)
            x = Util.preprocess(img)
            batch_x[i] = x
            if self.teacher is not None:
                tx = None
                if self.input_shape != self.teacher.input_shape[1:]:
                    tw = self.teacher.input_shape[1:][1]
                    th = self.teacher.input_shape[1:][0]
                    tx = Util.preprocess(Util.resize(img, (tw, th)))
                    batch_tx[i] = tx
            else:
                labeled_boxes = self.convert_to_boxes(label_lines)
                self.build_batch_tensor(labeled_boxes, batch_y, batch_mask, i)
            if self.debug:
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)
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

