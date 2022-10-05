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


def __smooth(y_true, alpha, true_only=False):
    if true_only:
        return K.clip(y_true, 0.0, 1.0 - alpha)
    else:
        return K.clip(y_true, 0.0 + alpha, 1.0 - alpha)


def __abs_log_loss(y_true, y_pred):
    return -K.log((1.0 + K.epsilon()) - K.abs(y_true - y_pred))


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    p_t = tf.where(K.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1.0), alpha_factor, 1.0 - alpha_factor)
    cross_entropy = K.binary_crossentropy(y_true, y_pred)
    weight = alpha_t * K.pow((1.0 - p_t), gamma)
    loss = weight * cross_entropy
    return loss


def __confidence_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    # loss = K.binary_crossentropy(obj_true, obj_pred)
    loss = focal_loss(obj_true, obj_pred)
    loss = K.mean(loss, axis=0)
    loss = K.sum(loss)
    return loss


def __iou(y_true, y_pred, diou=False):
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

    if diou:
        cxy_true = y_true[:, :, :, 1:3]
        cxy_pred = y_true[:, :, :, 1:3]
        center_distance = K.sum(K.square(cxy_true - cxy_pred), axis=-1)
        union_width = K.maximum(x2_true, x2_pred) - K.minimum(x1_true, x1_pred)
        union_height = K.maximum(y2_true, y2_pred) - K.minimum(y1_true, y1_pred)
        diagonal = K.square(union_width) + K.square(union_height) + K.epsilon()
        diou_factor = center_distance / diagonal
        diou_factor = K.mean(diou_factor, axis=0)
        diou_factor = K.sum(diou_factor)
        return iou, diou_factor
    else:
        return iou, 0.0


def __bbox_loss_xywh(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(K.sum(obj_true))
    if K.equal(obj_count, K.constant(0.0)):
        return 0.0

    # weight_mask = ((obj_true * 1.05) - (__iou(y_true, y_pred) * obj_true)) * 5.0
    weight_mask = 5.0
    xy_true = y_true[:, :, :, 1:3]
    xy_pred = y_pred[:, :, :, 1:3]
    xy_loss = __abs_log_loss(xy_true, xy_pred)
    xy_loss = K.sum(xy_loss, axis=-1) * weight_mask
    xy_loss = K.mean(xy_loss, axis=0)
    xy_loss = K.sum(xy_loss)

    eps = K.epsilon()
    wh_true = K.sqrt(y_true[:, :, :, 3:5] + eps)
    wh_pred = K.sqrt(y_pred[:, :, :, 3:5] + eps)
    wh_loss = __abs_log_loss(wh_true, wh_pred)
    wh_loss = K.sum(wh_loss, axis=-1) * weight_mask
    wh_loss = K.mean(wh_loss, axis=0)
    wh_loss = K.sum(wh_loss)
    return xy_loss + wh_loss


def __bbox_loss_iou(y_true, y_pred, ignore_threshold):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(K.sum(obj_true))
    if K.equal(obj_count, K.constant(0.0)):
        return 0.0

    iou, diou_factor = __iou(y_true, y_pred, diou=True)
    ignore_mask = tf.where(iou > ignore_threshold, 0.0, 1.0) * obj_true
    loss = obj_true - (iou * obj_true)
    loss = K.mean(loss * ignore_mask, axis=0)
    loss = K.sum(loss)
    return loss + diou_factor


def __bbox_loss(y_true, y_pred, ignore_threshold):
    return __bbox_loss_iou(y_true, y_pred, ignore_threshold=ignore_threshold)
    # return __bbox_loss_xywh(y_true, y_pred)


def __classification_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(K.sum(obj_true))
    if K.equal(obj_count, K.constant(0.0)):
        return 0.0

    # class_true = K.clip(y_true[:, :, :, 5:], 0.1, 0.9)
    class_true = y_true[:, :, :, 5:]
    class_pred = y_pred[:, :, :, 5:]
    # loss = K.binary_crossentropy(class_true, class_pred)
    loss = focal_loss(class_true, class_pred)
    # loss = __abs_log_loss(class_true, class_pred)
    loss = K.sum(loss, axis=-1) * obj_true
    loss = K.mean(loss, axis=0)
    loss = K.sum(loss)
    return loss


def confidence_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred)


def confidence_with_bbox_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred, ignore_threshold)


def yolo_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred, ignore_threshold) + __classification_loss(y_true, y_pred)
