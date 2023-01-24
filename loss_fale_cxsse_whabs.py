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


def __confidence_loss(y_true, y_pred, mask, alpha, gamma):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    ale = AbsoluteLogarithmicError(alpha=alpha, gamma=gamma)
    loss = tf.reduce_sum(ale(obj_true, obj_pred) * mask[:, :, :, 0])
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def __bbox_loss(y_true, y_pred, mask):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), y_pred.dtype)
    if obj_count == tf.constant(0.0):
        return 0.0

    y_true_shape = tf.cast(tf.shape(y_true), y_pred.dtype)
    grid_height, grid_width = y_true_shape[1], y_true_shape[2]

    cx_true = y_true[:, :, :, 1]
    cy_true = y_true[:, :, :, 2]

    x_range = tf.range(grid_width, dtype=y_pred.dtype)
    x_offset = tf.broadcast_to(x_range, shape=tf.shape(cx_true))

    y_range = tf.range(grid_height, dtype=y_pred.dtype)
    y_range = tf.reshape(y_range, shape=(1, grid_height, 1))
    y_offset = tf.broadcast_to(y_range, shape=tf.shape(cy_true))

    cx_true = x_offset + (cx_true * 1.0 / grid_width)
    cy_true = y_offset + (cy_true * 1.0 / grid_height)

    cx_pred = y_pred[:, :, :, 1]
    cy_pred = y_pred[:, :, :, 2]

    cx_pred = x_offset + (cx_pred * 1.0 / grid_width)
    cy_pred = y_offset + (cy_pred * 1.0 / grid_height)

    eps = tf.keras.backend.epsilon()
    w_true = tf.sqrt(y_true[:, :, :, 3] + eps)
    h_true = tf.sqrt(y_true[:, :, :, 4] + eps)
    w_pred = tf.sqrt(y_pred[:, :, :, 3] + eps)
    h_pred = tf.sqrt(y_pred[:, :, :, 4] + eps)

    xy_loss = tf.square(cx_true - cx_pred) + tf.square(cy_true - cy_pred)
    wh_loss = tf.abs(w_true - w_pred) + tf.abs(h_true - h_pred) * 0.05
    loss = tf.reduce_sum((xy_loss + wh_loss) * obj_true * mask[:, :, :, 0])
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def __classification_loss(y_true, y_pred, mask, alpha, gamma, label_smoothing):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), y_pred.dtype)
    if obj_count == tf.constant(0.0):
        return 0.0

    class_true = y_true[:, :, :, 5:]
    class_pred = y_pred[:, :, :, 5:]
    ale = AbsoluteLogarithmicError(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
    loss = tf.reduce_sum(tf.reduce_sum(ale(class_true, class_pred), axis=-1) * obj_true * mask[:, :, :, 0])
    return loss / tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)


def confidence_loss(y_true, y_pred, mask, gamma):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, mask, alpha, gamma)


def confidence_with_bbox_loss(y_true, y_pred, mask, gamma):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, mask, alpha, gamma) + __bbox_loss(y_true, y_pred, mask)


def yolo_loss(y_true, y_pred, mask, alpha, gamma, label_smoothing):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, mask, alpha, gamma) + __bbox_loss(y_true, y_pred, mask) + __classification_loss(y_true, y_pred, mask, alpha, gamma, label_smoothing)

