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
import numpy as np
import tensorflow as tf

from ace import AdaptiveCrossentropy as ACE
from tensorflow.python.framework.ops import convert_to_tensor_v2


def _obj_loss(y_true, y_pred, pos_mask, extra, iou, iou_obj_target, eps):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]

    neg_mask = tf.ones_like(pos_mask) - pos_mask

    num_pos = tf.reduce_sum(pos_mask)
    num_neg = tf.reduce_sum(neg_mask)
    if iou_obj_target == 1.0:
        obj_true = tf.clip_by_value(iou, 0.1, 1.0) * pos_mask

    obj_pos_loss = 0.0
    obj_neg_loss = 0.0
    if num_pos > 0.0:
        obj_pos_loss = tf.reduce_sum(ACE()(obj_true, obj_pred) * pos_mask)
    if num_neg > 0.0:
        obj_neg_loss = tf.reduce_sum(ACE()(obj_true, obj_pred) * neg_mask)
    return obj_pos_loss, obj_neg_loss, num_pos, num_neg


def _box_loss(y_true, y_pred, pos_mask, box_weight, loss_type='ciou'):
    num_pos = tf.reduce_sum(pos_mask)
    iou = tf.zeros_like(pos_mask)
    if num_pos == 0.0:
        return 0.0, iou

    y_true_shape = tf.cast(tf.shape(y_true), y_pred.dtype)
    rows, cols = y_true_shape[1], y_true_shape[2]

    x_grid, y_grid = tf.meshgrid(tf.range(cols), tf.range(rows), indexing='xy')

    cx_true = y_true[:, :, :, 1]
    cy_true = y_true[:, :, :, 2]
    cx_pred = y_pred[:, :, :, 1]
    cy_pred = y_pred[:, :, :, 2]

    cx_true = (x_grid + cx_true) / cols
    cx_pred = (x_grid + cx_pred) / cols
    cy_true = (y_grid + cy_true) / rows
    cy_pred = (y_grid + cy_pred) / rows

    w_true = y_true[:, :, :, 3]
    h_true = y_true[:, :, :, 4]
    w_pred = y_pred[:, :, :, 3]
    h_pred = y_pred[:, :, :, 4]

    if loss_type in ['l1', 'l2']:
        if loss_type == 'l1':
            cx_loss = tf.abs(cx_true - cx_pred)
            cy_loss = tf.abs(cy_true - cy_pred)
            w_loss = tf.abs(w_true - w_pred)
            h_loss = tf.abs(h_true - h_pred)
        else:
            cx_loss = tf.square(cx_true - cx_pred)
            cy_loss = tf.square(cy_true - cy_pred)
            w_loss = tf.square(w_true - w_pred)
            h_loss = tf.square(h_true - h_pred)
        loss = cx_loss + cy_loss + w_loss + h_loss
    else:
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

        intersection_w = tf.maximum(min_x2 - max_x1, 0.0)
        intersection_h = tf.maximum(min_y2 - max_y1, 0.0)
        intersection_area = intersection_w * intersection_h

        y_true_area = w_true * h_true
        y_pred_area = w_pred * h_pred
        union_area = y_true_area + y_pred_area - intersection_area
        iou = tf.clip_by_value(intersection_area / (union_area + 1e-5), 0.0, 1.0)

        if loss_type == 'iou':
            loss = pos_mask - iou
        else:
            convex_x1 = tf.minimum(x1_true, x1_pred)
            convex_y1 = tf.minimum(y1_true, y1_pred)
            convex_x2 = tf.maximum(x2_true, x2_pred)
            convex_y2 = tf.maximum(y2_true, y2_pred)

            convex_w = tf.maximum(convex_x2 - convex_x1, 0.0)
            convex_h = tf.maximum(convex_y2 - convex_y1, 0.0)

            center_distance = tf.square(cx_pred - cx_true) + tf.square(cy_pred - cy_true)
            convex_diagonal_length = tf.square(convex_w) + tf.square(convex_h)

            aspect_true = tf.math.atan(w_true / (h_true + 1e-5))
            aspect_pred = tf.math.atan(w_pred / (h_pred + 1e-5))
            v = 4.0 / (tf.square(tf.constant(np.pi))) * tf.square(aspect_true - aspect_pred)
            alpha = v / (1.0 - iou + v + 1e-5)

            ciou_term = (center_distance / (convex_diagonal_length + 1e-5)) + (alpha * v)
            loss = pos_mask - iou + ciou_term

    loss = tf.reduce_sum(loss * pos_mask)
    return loss * box_weight, iou


def _cls_loss(y_true, y_pred, pos_mask, extra, label_smoothing, eps):
    num_pos = tf.reduce_sum(pos_mask)
    if num_pos == 0.0:
        return 0.0

    cls_true = y_true[:, :, :, 5:]
    cls_pred = y_pred[:, :, :, 5:]
    cls_weight = extra[:, :, :, 5:]

    loss = tf.reduce_sum(tf.reduce_sum(ACE(label_smoothing=label_smoothing)(cls_true, cls_pred) * cls_weight, axis=-1) * pos_mask)
    return loss


def sbd_loss(y_true, y_pred, extra, iou_obj_target, box_weight, label_smoothing, eps=1e-7):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    pos_mask = tf.where(y_true[:, :, :, 0] == 1.0, 1.0, 0.0)
    box_loss, iou = _box_loss(y_true, y_pred, pos_mask, box_weight)
    obj_pos_loss, obj_neg_loss, num_pos, num_neg = _obj_loss(y_true, y_pred, pos_mask, extra, iou, iou_obj_target, eps)
    cls_loss = _cls_loss(y_true, y_pred, pos_mask, extra, label_smoothing, eps)
    return obj_pos_loss, obj_neg_loss, num_pos, num_neg, box_loss, cls_loss

