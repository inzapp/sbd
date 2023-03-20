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
import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from util import ModelUtil
from concurrent.futures.thread import ThreadPoolExecutor



class YoloDataGenerator:
    def __init__(self, image_paths, input_shape, output_shape, batch_size, multi_classification_at_same_box, ignore_nearby_cell, nearby_cell_ignore_threshold):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.input_width, self.input_height, self.input_channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        self.output_shapes = output_shape
        if type(self.output_shapes) == tuple:
            self.output_shapes = [self.output_shapes]
        self.num_classes = self.output_shapes[0][-1] - 5
        self.batch_size = batch_size
        self.num_output_layers = len(self.output_shapes)
        self.multi_classification_at_same_box = multi_classification_at_same_box
        self.ignore_nearby_cell = ignore_nearby_cell
        self.nearby_cell_ignore_threshold = nearby_cell_ignore_threshold 
        self.virtual_anchor_ws = []
        self.virtual_anchor_hs = []
        self.batch_index = 0
        self.pool = ThreadPoolExecutor(8)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.3),
            A.GaussianBlur(p=0.5, blur_limit=(5, 5))
        ])

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def is_label_exists(self, label_path):
        is_label_exists = False
        if os.path.exists(label_path) and os.path.isfile(label_path):
            is_label_exists = True
        return is_label_exists, label_path

    def check_labels_exist(self):
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.is_label_exists, f'{path[:-4]}.txt'))
        not_found_label_paths = set()
        for f in tqdm(fs):
            label_exists, label_path = f.result()
            if not label_exists:
                not_found_label_paths.add(label_path)
        if len(not_found_label_paths) > 0:
            print()
            for label_path in list(not_found_label_paths):
                print(f'label not found : {label_path}')
            ModelUtil.print_error_exit(f'{len(not_found_label_paths)} labels not found')
        print('label exist check success')

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
            ModelUtil.print_error_exit(f'{len(invalid_label_paths)} invalid label exists fix it')
        print('invalid label check success')

    def get_iou_with_virtual_anchors(self, box):
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
            iou = ModelUtil.iou(labeled_box, virtual_anchor_box)
            iou_with_virtual_anchors.append([layer_index, iou])
        return sorted(iou_with_virtual_anchors, key=lambda x: x[1], reverse=True)

    def calculate_class_weights(self):
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, f'{path[:-4]}.txt'))

        class_counts = np.zeros(shape=(self.num_classes,), dtype=np.int32)
        labeled_boxes, ws, hs = [], [], []
        for f in tqdm(fs):
            lines, label_path = f.result()
            for line in lines:
                class_index, cx, cy, w, h = list(map(float, line.split()))
                class_counts[int(class_index)] += 1
                labeled_boxes.append([cx, cy, w, h])
                ws.append(w)
                hs.append(h)

        class_counts_over_zero = class_counts[class_counts > 0]
        if class_counts_over_zero is None:
            class_weights = np.ones(shape=(self.num_classes,), dtype=np.float32)
        else:
            min_class_count = np.min(class_counts_over_zero) + 1e-5
            class_weights = np.asarray([min_class_count / class_count for class_count in class_counts], dtype=np.float32)
        
        class_weights_sum = np.sum(class_weights) + 1e-5
        if class_weights_sum > 4.0:
            print(class_weights)
            print(class_weights_sum)
            print()
            class_weights *= (4.0 / class_weights_sum)

        print(f'\nclass count, weight')
        for i in range(len(class_weights)):
            print(f'class {i:>3} : {class_counts[i]:>8}, {class_weights[i]:.2f}')

        # print(len(class_counts))
        # print(np.sum(class_weights))
        return class_weights

    def calculate_virtual_anchor(self):
        fs = []
        for path in self.image_paths:
            fs.append(self.pool.submit(self.load_label, f'{path[:-4]}.txt'))

        ignore_box_count = 0
        labeled_boxes, ws, hs = [], [], []
        for f in tqdm(fs):
            lines, label_path = f.result()
            for line in lines:
                class_index, cx, cy, w, h = list(map(float, line.split()))
                is_w_valid = int(w * self.input_shape[1]) > 3
                if is_w_valid:
                    ws.append(w)
                is_h_valid = int(h * self.input_shape[0]) > 3
                if is_h_valid:
                    hs.append(h)
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

        print('\naverage IoU with virtual anchors')
        best_iou_sum = 0.0
        for box in tqdm(labeled_boxes):
            iou_with_virtual_anchors = self.get_iou_with_virtual_anchors(box)
            best_iou = iou_with_virtual_anchors[0][1]
            best_iou_sum += best_iou
        avg_iou_with_virtual_anchor = best_iou_sum / (float(len(labeled_boxes)) + 1e-5)
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
            _, _, allocated_count = self.build_batch_tensor(labeled_boxes, virtual_anchor_training=False)
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
        return sorted(labeled_boxes, key=lambda x: x['area'], reverse=True)

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
                box_nearby = np.clip(np.array(box_nearby), 0.0, 1.0)
                iou = ModelUtil.iou(box_origin, box_nearby)
                nearby_cells.append({
                    'offset_y': offset_y,
                    'offset_x': offset_x,
                    'cx_grid': cx_nearby_grid,
                    'cy_grid': cy_nearby_grid,
                    'iou': iou})
        return sorted(nearby_cells, key=lambda x: x['iou'], reverse=True)

    def get_nearby_grids_for_mask(self, confidence_channel, rows, cols, row, col, cx_grid, cy_grid, cx_raw, cy_raw, w, h, offset_range=1):
        assert offset_range > 0
        nearby_cells = []
        offset_vals = list(range(-offset_range, offset_range))
        positions = []
        for offset_i in offset_vals:
            for offset_j in offset_vals:
                positions.append([offset_i, offset_j])
        for offset_y, offset_x in positions:
            if (0 <= row + offset_y < rows) and (0 <= col + offset_x < cols):
                box_origin = [
                    cx_raw - (w * 0.5),
                    cy_raw - (h * 0.5),
                    cx_raw + (w * 0.5),
                    cy_raw + (h * 0.5)]
                cx_nearby_raw = (float(col + offset_x) + 0.5) / float(cols)
                cy_nearby_raw = (float(row + offset_y) + 0.5) / float(rows)
                box_nearby = [
                    cx_nearby_raw - (w * 0.5),
                    cy_nearby_raw - (h * 0.5),
                    cx_nearby_raw + (w * 0.5),
                    cy_nearby_raw + (h * 0.5)]
                box_nearby = np.clip(np.array(box_nearby), 0.0, 1.0)
                iou = ModelUtil.iou(box_origin, box_nearby)
                nearby_cells.append({
                    'offset_y': offset_y,
                    'offset_x': offset_x,
                    'iou': iou})
        return sorted(nearby_cells, key=lambda x: x['iou'], reverse=True)

    def build_batch_tensor(self, labeled_boxes, virtual_anchor_training):
        y, mask = [], []
        for i in range(self.num_output_layers):
            y.append(np.zeros(shape=tuple(self.output_shapes[i][1:]), dtype=np.float32))
            mask.append(np.ones(shape=tuple(self.output_shapes[i][1:]), dtype=np.float32))

        allocated_count = 0
        for b in labeled_boxes:
            class_indexes, cx, cy, w, h = b['class_indexes'], b['cx'], b['cy'], b['w'], b['h']
            w_not_valid = int(w * self.input_shape[1]) <= 3
            h_not_valid = int(h * self.input_shape[0]) <= 3
            if w_not_valid and h_not_valid:
                continue

            best_iou_layer_indexes = self.get_iou_with_virtual_anchors([cx, cy, w, h])
            is_box_allocated = False
            for i, layer_iou in best_iou_layer_indexes:
                if is_box_allocated and layer_iou < 0.6:
                    break
                output_rows = float(self.output_shapes[i][1])
                output_cols = float(self.output_shapes[i][2])
                rr, cc = np.meshgrid(np.arange(output_rows), np.arange(output_cols), indexing='ij')
                rr = np.asarray(rr).astype('float32') / np.max(rr)
                cc = np.asarray(cc).astype('float32') / np.max(cc)
                center_row = int(cy * output_rows)
                center_col = int(cx * output_cols)
                cx_grid_scale = (cx - float(center_col) / output_cols) / (1.0 / output_cols)
                cy_grid_scale = (cy - float(center_row) / output_rows) / (1.0 / output_rows)
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
                    offset_center_row = center_row + offset_y
                    offset_center_col = center_col + offset_x
                    if y[i][offset_center_row][offset_center_col][0] == 0.0:
                        y[i][offset_center_row][offset_center_col][0] = 1.0
                        if virtual_anchor_training:
                            y[i][:, :, 1] = cx_grid
                            y[i][:, :, 2] = cy_grid
                            y[i][:, :, 3] = self.virtual_anchor_ws[i]
                            y[i][:, :, 4] = self.virtual_anchor_hs[i]
                        else:
                            y[i][offset_center_row][offset_center_col][1] = cx_grid
                            y[i][offset_center_row][offset_center_col][2] = cy_grid
                            y[i][offset_center_row][offset_center_col][3] = w
                            y[i][offset_center_row][offset_center_col][4] = h
                        for class_index in class_indexes:
                            class_channel = y[i][:, :, class_index+5]
                            gaussian_segmentation = 1.0 - np.clip((np.abs(rr - cy) / (h * 0.5)) ** 2 + (np.abs(cc - cx) / (w * 0.5)) ** 2, 0.0, 1.0) ** 0.2
                            segmentation_indexes = np.where(gaussian_segmentation > class_channel)
                            class_channel[segmentation_indexes] = gaussian_segmentation[segmentation_indexes]
                            y[i][center_row][center_col][class_index+5] = 1.0
                        is_box_allocated = True
                        allocated_count += 1
                        break
        # for i in range(self.output_shapes[0][1]):
        #     for j in range(self.output_shapes[0][2]):
        #         print(f'{int(y[0][i][j][0])} ', end='')
        #     print()
        # exit(0)

        # for i in range(len(y)):
        #     for class_index in range(self.num_classes):
        #         cv2.imshow(f'class_{class_index}', cv2.resize(np.asarray(y[i][:, :, class_index+5] * 255.0).astype('uint8'), (0, 0), fx=8, fy=8))
        # key = cv2.waitKey(0)
        # if key == 27:
        #     exit(0)

        # create mask after all value is allocated in train tensor
        if self.ignore_nearby_cell:
            for b in labeled_boxes:
                class_indexes, cx, cy, w, h = b['class_indexes'], b['cx'], b['cy'], b['w'], b['h']
                best_iou_layer_indexes = self.get_iou_with_virtual_anchors([cx, cy, w, h])
                for i, layer_iou in best_iou_layer_indexes:
                    output_rows = float(self.output_shapes[i][1])
                    output_cols = float(self.output_shapes[i][2])
                    center_row = int(cy * output_rows)
                    center_col = int(cx * output_cols)
                    cx_grid_scale = (cx - float(center_col) / output_cols) / (1.0 / output_cols)
                    cy_grid_scale = (cy - float(center_row) / output_rows) / (1.0 / output_rows)
                    nearby_grids = self.get_nearby_grids_for_mask(
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
                        if grid['iou'] < self.nearby_cell_ignore_threshold:
                            break
                        offset_y = grid['offset_y']
                        offset_x = grid['offset_x']
                        offset_center_row = center_row + offset_y
                        offset_center_col = center_col + offset_x
                        # if y[i][offset_center_row][offset_center_col][0] == 1.0:  # debug mark object for 2
                        #     mask[i][offset_center_row][offset_center_col][0] = 2.0
                        if offset_y == 0 and offset_x == 0:  # ignore allocated object
                            continue
                        if y[i][offset_center_row][offset_center_col][0] == 0.0:
                            mask[i][offset_center_row][offset_center_col][0] = 0.0

            # for i in range(self.output_shapes[0][1]):
            #     for j in range(self.output_shapes[0][2]):
            #         print(f'{int(mask[0][i][j][0])} ', end='')
            #     print()
            # exit(0)
        return y, mask, allocated_count
            
    def load(self, virtual_anchor_training=False):
        fs, batch_x, batch_y, batch_mask = [], [], [], []
        for i in range(self.num_output_layers):
            batch_y.append([])
            batch_mask.append([])
        for path in self.get_next_batch_image_paths():
            fs.append(self.pool.submit(ModelUtil.load_img, path, self.input_channel))
        for f in fs:
            img, _, cur_img_path = f.result()
            # cv2.imshow('img', img)
            img = self.transform(image=img)['image']
            img = ModelUtil.resize(img, (self.input_width, self.input_height))
            x = ModelUtil.preprocess(img)
            batch_x.append(x)

            with open(f'{cur_img_path[:-4]}.txt', mode='rt') as file:
                label_lines = file.readlines()
            labeled_boxes = self.convert_to_boxes(label_lines)
            np.random.shuffle(labeled_boxes)

            y, mask, _ = self.build_batch_tensor(labeled_boxes, virtual_anchor_training)
            for i in range(self.num_output_layers):
                batch_y[i].append(y[i])
                batch_mask[i].append(mask[i])
        batch_x = np.asarray(batch_x).astype('float32')
        for i in range(self.num_output_layers):
            batch_y[i] = np.asarray(batch_y[i]).astype('float32')
            batch_mask[i] = np.asarray(batch_mask[i]).astype('float32')

        if self.num_output_layers == 1:
            return batch_x, batch_y[0], batch_mask[0]
        else:
            return batch_x, batch_y, batch_mask

    def get_next_batch_image_paths(self):
        start_index = self.batch_size * self.batch_index
        end_index = start_index + self.batch_size
        batch_image_paths = self.image_paths[start_index:end_index]
        self.batch_index += 1
        if self.batch_index == self.__len__():
            self.batch_index = 0
            np.random.shuffle(self.image_paths)
        return batch_image_paths

