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
import tensorflow as tf

from ale import AbsoluteLogarithmicError
from tensorflow.python.framework.ops import convert_to_tensor_v2


def __confidence_loss(y_true, y_pred, mask, alpha, gamma):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    ale = AbsoluteLogarithmicError(alpha=alpha, gamma=gamma)
    loss = tf.reduce_sum(ale(obj_true, obj_pred) * mask[:, :, :, 0])
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def __iou(y_true, y_pred, convex=False):
    y_true_shape = tf.cast(tf.shape(y_true), y_pred.dtype)
    grid_height, grid_width = y_true_shape[1], y_true_shape[2]

    x_grid, y_grid = tf.meshgrid(tf.range(grid_width), tf.range(grid_height), indexing='xy')

    cx_true = y_true[:, :, :, 1]
    cy_true = y_true[:, :, :, 2]
    cx_pred = y_pred[:, :, :, 1]
    cy_pred = y_pred[:, :, :, 2]

    cx_true = (x_grid + cx_true) / grid_width
    cx_pred = (x_grid + cx_pred) / grid_width
    cy_true = (y_grid + cy_true) / grid_height
    cy_pred = (y_grid + cy_pred) / grid_height

    w_true = y_true[:, :, :, 3]
    h_true = y_true[:, :, :, 4]
    w_pred = y_pred[:, :, :, 3]
    h_pred = y_pred[:, :, :, 4]

    x1_true = cx_true - (w_true * 0.5)
    y1_true = cy_true - (h_true * 0.5)
    x2_true = cx_true + (w_true * 0.5)
    y2_true = cy_true + (h_true * 0.5)

    x1_pred = cx_pred - (w_pred * 0.5)
    y1_pred = cy_pred - (h_pred * 0.5)
    x2_pred = cx_pred + (w_pred * 0.5)
    y2_pred = cy_pred + (h_pred * 0.5)

    min_x2 = tf.minimum(x2_true, x2_pred)
    max_x1 = tf.maximum(x1_true, x1_pred)
    min_y2 = tf.minimum(y2_true, y2_pred)
    max_y1 = tf.maximum(y1_true, y1_pred)

    intersection_width = tf.maximum(min_x2 - max_x1, 0.0)
    intersection_height = tf.maximum(min_y2 - max_y1, 0.0)
    intersection = intersection_width * intersection_height

    convex_width = tf.maximum(x2_true, x2_pred) - tf.minimum(x1_true, x1_pred)
    convex_height = tf.maximum(y2_true, y2_pred) - tf.minimum(y1_true, y1_pred)

    y_true_area = w_true * h_true
    y_pred_area = w_pred * h_pred
    if convex:
        union = convex_width * convex_height
    else:
        union = y_true_area + y_pred_area - intersection
    iou = intersection / (union + 1e-5)
    return iou


def __bbox_loss(y_true, y_pred, mask):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), y_pred.dtype)
    if obj_count == tf.constant(0.0):
        return 0.0

    iou = __iou(y_true, y_pred, convex=True)
    loss = tf.reduce_sum((obj_true - iou) * obj_true)
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def __classification_loss(y_true, y_pred, mask, alpha, gamma, label_smoothing):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), y_pred.dtype)
    if obj_count == tf.constant(0.0):
        return 0.0

    class_true = y_true[:, :, :, 5:]
    class_pred = y_pred[:, :, :, 5:]
    ale = AbsoluteLogarithmicError(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
    loss = tf.reduce_sum(tf.reduce_sum(ale(class_true, class_pred), axis=-1) * obj_true)
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def confidence_loss(y_true, y_pred, mask, obj_alpha, obj_gamma, label_smoothing):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, mask, obj_alpha, obj_gamma), -1.0, -1.0


def confidence_with_bbox_loss(y_true, y_pred, mask, obj_alpha, obj_gamma, label_smoothing):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, mask, obj_alpha, obj_gamma), __bbox_loss(y_true, y_pred, mask), -1.0


def sbd_loss(y_true, y_pred, mask, obj_alpha, obj_gamma, cls_alpha, cls_gamma, label_smoothing):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, mask, obj_alpha, obj_gamma), __bbox_loss(y_true, y_pred, mask), __classification_loss(y_true, y_pred, mask, cls_alpha, cls_gamma, label_smoothing)

