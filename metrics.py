"""
Authors : inzapp
Github url : https://github.com/inzapp/yolo

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
import tensorflow as tf


def precision(y_true, y_pred):
    """
    precision = True Positive / True Positive + False Positive
    precision = True Positive / All Detections
    """
    tp = tf.reduce_sum(y_pred[:, :, :, 0] * y_true[:, :, :, 0])
    return tp / (tf.reduce_sum(y_pred[:, :, :, 0]) + 1e-5)


def recall(y_true, y_pred):
    """
    recall = True Positive / True Positive + False Negative
    recall = True Positive / All Ground Truths
    """
    tp = tf.reduce_sum(y_pred[:, :, :, 0] * y_true[:, :, :, 0])
    return tp / (tf.reduce_sum(y_true[:, :, :, 0]) + 1e-5)


def f1(y_true, y_pred):
    """
    Harmonic mean of precision and recall.
    f1 = 1 / (precision^-1 + precision^-1) * 0.5
    f1 = 2 * precision * recall / (precision + recall)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (p * r * 2.0) / (p + r + 1e-5)


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
    if intersection_width < 0.0 or intersection_height < 0.0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
    b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
    union_area = a_area + b_area - intersection_area
    return intersection_area / float(union_area)


def iou_f1(y_true, y_pred):
    # TODO : 부하가 너무 커서 compile metric 에는 사용을 못할거 같다. 다른 방식으로 해보자
    confidence_threshold = 0.25
    iou_threshold = 0.5
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    bbox_true = []
    for batch_index in range(len(y_true)):
        for i in range(len(y_true[batch_index])):
            for j in range(len(y_true[batch_index][i])):
                if y_true[batch_index][i][j][0] == 1.0:
                    cx = y_true[batch_index][i][j][1]
                    cy = y_true[batch_index][i][j][2]
                    w = y_true[batch_index][i][j][3]
                    h = y_true[batch_index][i][j][4]
                    x1, y1 = cx - w / 2.0, cy - h / 2.0
                    x2, y2 = cx + w / 2.0, cy + h / 2.0
                    x1 = int(x1 * 1000.0)
                    y1 = int(y1 * 1000.0)
                    x2 = int(x2 * 1000.0)
                    y2 = int(y2 * 1000.0)
                    class_index = -1
                    max_value = 0.0
                    for channel_index in range(5, len(y_true[batch_index][i][j])):
                        if y_true[batch_index][i][j][channel_index] > max_value:
                            max_value = y_true[batch_index][i][j][channel_index]
                            class_index = channel_index - 5
                    bbox_true.append([x1, y1, x2, y2, class_index])

    bbox_pred = []
    for batch_index in range(len(y_pred)):
        for i in range(len(y_pred[batch_index])):
            for j in range(len(y_pred[batch_index][i])):
                if y_pred[batch_index][i][j][0] >= confidence_threshold:
                    cx = y_pred[batch_index][i][j][1]
                    cy = y_pred[batch_index][i][j][2]
                    w = y_pred[batch_index][i][j][3]
                    h = y_pred[batch_index][i][j][4]
                    x1, y1 = cx - w / 2.0, cy - h / 2.0
                    x2, y2 = cx + w / 2.0, cy + h / 2.0
                    x1 = int(x1 * 1000.0)
                    y1 = int(y1 * 1000.0)
                    x2 = int(x2 * 1000.0)
                    y2 = int(y2 * 1000.0)
                    class_index = -1
                    max_value = 0.0
                    for channel_index in range(5, len(y_pred[batch_index][i][j])):
                        if y_pred[batch_index][i][j][channel_index] > max_value:
                            max_value = y_pred[batch_index][i][j][channel_index]
                            class_index = channel_index - 5
                    bbox_pred.append([x1, y1, x2, y2, class_index])

    tp = 0
    for cur_bbox_true in bbox_true:
        for cur_bbox_pred in bbox_pred:
            if iou(cur_bbox_true[:4], cur_bbox_pred[:4]) > iou_threshold:
                tp += 1
                break

    fp = 0
    for cur_bbox_pred in bbox_pred:
        found = False
        for cur_bbox_true in bbox_true:
            if iou(cur_bbox_true[:4], cur_bbox_pred[:4]) > iou_threshold:
                found = True
                break
        if not found:
            fp += 1

    fn = 0
    for cur_bbox_true in bbox_true:
        found = False
        for cur_bbox_pred in bbox_pred:
            if iou(cur_bbox_true[:4], cur_bbox_pred[:4]) > iou_threshold:
                found = True
                break
        if not found:
            fn += 1

    precision = tp / float(tp + fp + 1e-5)
    recall = tp / float(tp + fn + 1e-5)
    return (precision * recall * 2.0) / (precision + recall + 1e-5)
