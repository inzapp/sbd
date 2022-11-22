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
from tensorflow.python.framework.ops import convert_to_tensor_v2
from ale import AbsoluteLogarithmicError


def __confidence_loss(y_true, y_pred, alpha, gamma):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    loss = AbsoluteLogarithmicError()(obj_true, obj_pred)
    obj_count = tf.cast(tf.reduce_sum(obj_true), dtype=loss.dtype) + tf.keras.backend.epsilon()
    obj_loss = (loss * obj_true) / obj_count
    background_loss = AbsoluteLogarithmicError(alpha=0.5, gamma=1.5)(obj_true, obj_pred) * (1.0 - obj_true) * 2.0
    loss = tf.reduce_sum(obj_loss + background_loss)
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def __iou(y_true, y_pred):
    y_true_shape = tf.cast(tf.shape(y_true), y_pred.dtype)
    grid_height, grid_width = y_true_shape[1], y_true_shape[2]

    cx_true = y_true[:, :, :, 1]
    cy_true = y_true[:, :, :, 2]
    w_true = y_true[:, :, :, 3]
    h_true = y_true[:, :, :, 4]

    x_range = tf.range(grid_width, dtype=y_pred.dtype)
    x_offset = tf.broadcast_to(x_range, shape=tf.shape(cx_true))

    y_range = tf.range(grid_height, dtype=y_pred.dtype)
    y_range = tf.reshape(y_range, shape=(1, grid_height, 1))
    y_offset = tf.broadcast_to(y_range, shape=tf.shape(cy_true))

    cx_true = x_offset + (cx_true * 1.0 / grid_width)
    cy_true = y_offset + (cy_true * 1.0 / grid_height)

    x1_true = cx_true - (w_true * 0.5)
    y1_true = cy_true - (h_true * 0.5)
    x2_true = cx_true + (w_true * 0.5)
    y2_true = cy_true + (h_true * 0.5)

    cx_pred = y_pred[:, :, :, 1]
    cy_pred = y_pred[:, :, :, 2]
    w_pred = y_pred[:, :, :, 3]
    h_pred = y_pred[:, :, :, 4]

    cx_pred = x_offset + (cx_pred * 1.0 / grid_width)
    cy_pred = y_offset + (cy_pred * 1.0 / grid_height)

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

    y_true_area = w_true * h_true
    y_pred_area = w_pred * h_pred
    union = y_true_area + y_pred_area - intersection
    iou = intersection / (union + 1e-5)
    return iou


def __bbox_loss(y_true, y_pred, alpha, gamma):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), dtype=y_pred.dtype)
    if obj_count == tf.constant(0.0):
        return 0.0

    xy_true = y_true[:, :, :, 1:3]
    xy_pred = y_pred[:, :, :, 1:3]
    xy_loss = AbsoluteLogarithmicError()(xy_true, xy_pred)

    eps = tf.keras.backend.epsilon()
    wh_true = tf.sqrt(y_true[:, :, :, 3:5] + eps)
    wh_pred = tf.sqrt(y_pred[:, :, :, 3:5] + eps)
    wh_loss = AbsoluteLogarithmicError()(wh_true, wh_pred)

    loss = tf.reduce_sum(tf.reduce_sum(xy_loss + wh_loss, axis=-1) * obj_true)
    loss = (loss / obj_count / 4.0) * 5.0
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def __classification_loss(y_true, y_pred, alpha, gamma, label_smoothing):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), dtype=y_pred.dtype)
    if obj_count == tf.constant(0.0):
        return 0.0

    class_true = y_true[:, :, :, 5:]
    class_pred = y_pred[:, :, :, 5:]
    loss = AbsoluteLogarithmicError(label_smoothing=label_smoothing)(class_true, class_pred)
    num_classes = tf.cast(tf.shape(y_true)[-1], dtype=loss.dtype) - 5.0
    loss = tf.reduce_sum(tf.reduce_sum(loss, axis=-1) * obj_true)
    loss = loss / obj_count / num_classes
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def confidence_loss(y_true, y_pred, alpha, gamma):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, alpha, gamma)


def confidence_with_bbox_loss(y_true, y_pred, alpha, gamma):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, alpha, gamma) + __bbox_loss(y_true, y_pred, alpha, gamma)


def yolo_loss(y_true, y_pred, alpha, gamma, label_smoothing):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, alpha, gamma) + __bbox_loss(y_true, y_pred, alpha, gamma) + __classification_loss(y_true, y_pred, alpha, gamma, label_smoothing)

