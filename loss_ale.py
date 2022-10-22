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
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework.ops import convert_to_tensor_v2


def __ale(y_true, y_pred, focal=False, eps=1e-7):
    abs_error = tf.abs(y_true - y_pred)
    loss = -tf.math.log((1.0 + eps) - abs_error)
    if focal:
        loss *= abs_error
    return loss


def __confidence_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    loss = __ale(obj_true, obj_pred, focal=True)
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_sum(loss)
    return loss


def __iou(y_true, y_pred):
    y_true_shape = K.cast_to_floatx(K.shape(y_true))
    grid_height, grid_width = y_true_shape[1], y_true_shape[2]

    cx_true = y_true[:, :, :, 1]
    cy_true = y_true[:, :, :, 2]
    w_true = y_true[:, :, :, 3]
    h_true = y_true[:, :, :, 4]

    x_range = tf.range(grid_width, dtype=K.floatx())
    x_offset = tf.broadcast_to(x_range, shape=K.shape(cx_true))

    y_range = tf.range(grid_height, dtype=K.floatx())
    y_range = tf.reshape(y_range, shape=(1, grid_height, 1))
    y_offset = tf.broadcast_to(y_range, shape=K.shape(cy_true))

    cx_true = x_offset + (cx_true * 1.0 / grid_width)
    cy_true = y_offset + (cy_true * 1.0 / grid_height)

    x1_true = cx_true - w_true / 2.0
    y1_true = cy_true - h_true / 2.0
    x2_true = cx_true + w_true / 2.0
    y2_true = cy_true + h_true / 2.0

    cx_pred = y_pred[:, :, :, 1]
    cy_pred = y_pred[:, :, :, 2]
    w_pred = y_pred[:, :, :, 3]
    h_pred = y_pred[:, :, :, 4]

    cx_pred = x_offset + (cx_pred * 1.0 / grid_width)
    cy_pred = y_offset + (cy_pred * 1.0 / grid_height)

    x1_pred = cx_pred - w_pred / 2.0
    y1_pred = cy_pred - h_pred / 2.0
    x2_pred = cx_pred + w_pred / 2.0
    y2_pred = cy_pred + h_pred / 2.0

    min_x2 = K.minimum(x2_true, x2_pred)
    max_x1 = K.maximum(x1_true, x1_pred)
    min_y2 = K.minimum(y2_true, y2_pred)
    max_y1 = K.maximum(y1_true, y1_pred)

    intersection_width = K.maximum(min_x2 - max_x1, 0.0)
    intersection_height = K.maximum(min_y2 - max_y1, 0.0)
    intersection = intersection_width * intersection_height

    y_true_area = w_true * h_true
    y_pred_area = w_pred * h_pred
    union = y_true_area + y_pred_area - intersection
    iou = intersection / (union + 1e-5)
    return iou


def __bbox_loss(y_true, y_pred, ignore_threshold):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(tf.reduce_sum(obj_true))
    if obj_count == tf.constant(0.0):
        return 0.0

    xy_true = y_true[:, :, :, 1:3]
    xy_pred = y_pred[:, :, :, 1:3]
    xy_loss = __ale(xy_true, xy_pred)
    xy_loss = tf.reduce_sum(xy_loss, axis=-1) * obj_true
    xy_loss = tf.reduce_mean(xy_loss, axis=0)
    xy_loss = tf.reduce_sum(xy_loss)

    liou_loss = __ale(obj_true, __iou(y_true, y_pred) * obj_true)
    liou_loss = tf.reduce_mean(liou_loss, axis=0)
    liou_loss = tf.reduce_sum(liou_loss)
    return xy_loss + liou_loss


def __classification_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(tf.reduce_sum(obj_true))
    if obj_count == tf.constant(0.0):
        return 0.0

    class_true = y_true[:, :, :, 5:]
    class_pred = y_pred[:, :, :, 5:]
    loss = __ale(class_true, class_pred, focal=True)
    loss = tf.reduce_sum(loss, axis=-1) * obj_true
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_sum(loss)
    return loss


def confidence_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred)


def confidence_with_bbox_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred, ignore_threshold)


def yolo_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred, ignore_threshold) + __classification_loss(y_true, y_pred)

