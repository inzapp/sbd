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


def __smooth(y_true, alpha, true_only=False):
    if true_only:
        return tf.clip_by_value(y_true, 0.0, 1.0 - alpha)
    else:
        return tf.clip_by_value(y_true, 0.0 + alpha, 1.0 - alpha)


def __abs_log_loss(y_true, y_pred):
    return -tf.math.log((1.0 + tf.keras.backend.epsilon()) - tf.abs(y_true - y_pred))


def __confidence_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    loss = tf.keras.backend.binary_crossentropy(obj_true, obj_pred)
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_sum(loss)
    return loss


def __iou(y_true, y_pred):
    y_true_shape = tf.cast(tf.shape(y_true), dtype=tf.dtypes.float32)
    grid_height, grid_width = y_true_shape[1], y_true_shape[2]

    cx_true = y_true[:, :, :, 1]
    cy_true = y_true[:, :, :, 2]
    w_true = y_true[:, :, :, 3]
    h_true = y_true[:, :, :, 4]

    x_range = tf.range(grid_width, dtype=tf.dtypes.float32)
    x_offset = tf.broadcast_to(x_range, shape=tf.shape(cx_true))

    y_range = tf.range(grid_height, dtype=tf.dtypes.float32)
    y_range = tf.reshape(y_range, shape=(1, grid_height, 1))
    y_offset = tf.broadcast_to(y_range, shape=tf.shape(cy_true))

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
    return intersection / (union + 1e-5)


def __bbox_loss_xywh(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), tf.float32)
    if tf.equal(obj_count, tf.constant(0.0)):
        return 0.0

    weight_mask = (((obj_true + 0.05) * obj_true) - (__iou(y_true, y_pred) * obj_true)) * 5.0
    xy_true = y_true[:, :, :, 1:3]
    xy_pred = y_pred[:, :, :, 1:3]
    xy_loss = __abs_log_loss(xy_true, xy_pred)
    xy_loss = tf.reduce_sum(xy_loss, axis=-1) * weight_mask
    xy_loss = tf.reduce_mean(xy_loss, axis=0)
    xy_loss = tf.reduce_sum(xy_loss)

    eps = tf.keras.backend.epsilon()
    wh_true = tf.sqrt(y_true[:, :, :, 3:5] + eps)
    wh_pred = tf.sqrt(y_pred[:, :, :, 3:5] + eps)
    wh_loss = __abs_log_loss(wh_true, wh_pred)
    wh_loss = tf.reduce_sum(wh_loss, axis=-1) * weight_mask
    wh_loss = tf.reduce_mean(wh_loss, axis=0)
    wh_loss = tf.reduce_sum(wh_loss)
    return xy_loss + wh_loss


def __bbox_loss_iou(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), tf.float32)
    if tf.equal(obj_count, tf.constant(0.0)):
        return 0.0

    loss = tf.keras.backend.binary_crossentropy(obj_true, __iou(y_true, y_pred) * obj_true)
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_sum(loss)
    return loss


def __bbox_loss(y_true, y_pred):
    # return __bbox_loss_iou(y_true, y_pred)
    return __bbox_loss_xywh(y_true, y_pred)


def __classification_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = tf.cast(tf.reduce_sum(obj_true), tf.float32)
    if tf.equal(obj_count, tf.constant(0.0)):
        return 0.0

    y_true_shape = tf.shape(y_true)
    num_classes = tf.cast(y_true_shape[-1] - 5, tf.int32)
    obj_pred = y_pred[:, :, :, 0]
    expanded_obj_pred = tf.repeat(tf.expand_dims(obj_pred, axis=-1), num_classes, axis=-1)
    obj_pred_mask = tf.where(obj_pred > 0.0, 1.0, 0.0)

    b_obj_true = tf.cast(obj_true, tf.bool)
    b_obj_pred_mask = tf.cast(obj_pred_mask, tf.bool)
    fp_included_obj_true_mask = tf.cast(tf.logical_or(b_obj_true, b_obj_pred_mask), tf.float32)
    class_true = y_true[:, :, :, 5:]
    class_pred = y_pred[:, :, :, 5:] * expanded_obj_pred
    loss = tf.keras.backend.binary_crossentropy(class_true, class_pred)
    loss = tf.reduce_sum(loss, axis=-1) * fp_included_obj_true_mask
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_sum(loss)
    return loss


def confidence_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred)


def confidence_with_bbox_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred)


def yolo_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred) + __classification_loss(y_true, y_pred)
