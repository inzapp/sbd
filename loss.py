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


def __reverse_sigmoid(x):
    eps = tf.keras.backend.epsilon()
    t = tf.clip_by_value(x - eps, 0.0, 1.0)
    f = tf.clip_by_value(((x + eps) * -1.0 + eps * 2.0), 0.0, 1.0)
    x = t + f
    x = -tf.math.log(x / (1.0 - x))
    return x


def __abs_log_loss(y_true, y_pred):
    ones = 1.0 + tf.keras.backend.epsilon() * 10.0
    loss = -tf.math.log(ones - tf.abs(y_true - y_pred))
    return loss
    # return tf.square(y_true - y_pred)
    # return -tf.math.log(-tf.abs(y_true - y_pred) + 1.0 + (tf.keras.backend.epsilon() * 10.0))
    # return tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=False)


# def __abs_log_loss(y_true, y_pred, gamma=0.5):
#     # ones_like = 1.0 + tf.keras.backend.epsilon() * 10.0
#     ones_like = 1.0
#     x = tf.abs(y_true - y_pred)
#     loss = -tf.math.log(ones_like - x)
#     loss *= tf.math.pow(x, gamma)
#     return loss
# return tf.square(y_true - y_pred)
# return -tf.math.log(-tf.abs(y_true - y_pred) + 1.0 + (tf.keras.backend.epsilon() * 10.0))
# return tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=False)


def binary_focal_loss_fixed(y_true, y_pred):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma = 2.0
    alpha = 0.25
    eps = tf.keras.backend.epsilon()

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.keras.backend.clip(y_pred, eps, 1.0 - eps)
    p_t = tf.where(tf.keras.backend.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = tf.keras.backend.ones_like(y_true) * alpha
    alpha_t = tf.where(tf.keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -tf.keras.backend.log(p_t)
    weight = alpha_t * tf.keras.backend.pow((1 - p_t), gamma)
    loss = weight * cross_entropy
    # loss = tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=1))
    return loss


# def __confidence_loss_obj_only(y_true, y_pred):
#     obj_true = y_true[:, :, :, 0]
#     obj_pred = y_pred[:, :, :, 0]
#
#     # smooth_obj_true = __smooth(obj_true, alpha=0.02, true_only=True)  # bce conf loss
#     smooth_obj_true = __smooth(__iou(y_true, y_pred), alpha=0.02, true_only=True)  # iou conf loss
#     obj_confidence_loss = __loss(smooth_obj_true, obj_pred) * obj_true
#     obj_confidence_loss = tf.reduce_mean(obj_confidence_loss, axis=0)
#     obj_confidence_loss = tf.reduce_sum(obj_confidence_loss)
#     return obj_confidence_loss


def __zero_confidence_loss(y_true, y_pred):
    obj_pred = y_pred[:, :, :, 0]
    zeros = __smooth(tf.zeros(shape=tf.shape(obj_pred), dtype=tf.dtypes.float32), alpha=0.02)
    zero_loss = __abs_log_loss(zeros, obj_pred)
    zero_loss = tf.reduce_mean(zero_loss, axis=0)
    zero_loss = tf.reduce_sum(zero_loss)
    return zero_loss


# origin confidence loss
def __confidence_loss(y_true, y_pred):
    no_obj = 1.0
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    obj_false = tf.ones(shape=tf.shape(obj_true), dtype=tf.dtypes.float32) - obj_true

    smooth_obj_true = __smooth(obj_true, alpha=0.02, true_only=True)
    obj_confidence_loss = __abs_log_loss(smooth_obj_true, obj_pred) * obj_true  # log conf loss

    # smooth_iou = __smooth(__iou(y_true, y_pred), alpha=0.02, true_only=True)
    # obj_confidence_loss = __loss(smooth_iou, obj_pred) * obj_true  # iou conf loss

    obj_confidence_loss = tf.reduce_mean(obj_confidence_loss, axis=0)
    obj_confidence_loss = tf.reduce_sum(obj_confidence_loss)

    zeros = tf.zeros(shape=tf.shape(obj_true), dtype=tf.dtypes.float32)
    no_obj_confidence_loss = __abs_log_loss(zeros, obj_pred) * obj_false
    no_obj_confidence_loss = tf.reduce_mean(no_obj_confidence_loss, axis=0)
    no_obj_confidence_loss = tf.reduce_sum(no_obj_confidence_loss)
    return obj_confidence_loss + (no_obj_confidence_loss * no_obj)


def __iou(y_true, y_pred):
    y_true_shape = tf.cast(tf.shape(y_true), dtype=tf.dtypes.float32)
    grid_height, grid_width = y_true_shape[1], y_true_shape[2]
    eps = tf.keras.backend.epsilon()

    cx_true = y_true[:, :, :, 1]
    cy_true = y_true[:, :, :, 2]
    w_true = tf.sqrt(y_true[:, :, :, 3] + eps)
    h_true = tf.sqrt(y_true[:, :, :, 4] + eps)

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
    w_pred = tf.sqrt(y_pred[:, :, :, 3] + eps)
    h_pred = tf.sqrt(y_pred[:, :, :, 4] + eps)

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


# origin bbox loss
def __bbox_loss(y_true, y_pred):
    """
    BCE (sqrt(obj(x))) at width and height regression loss
    Sqrt was used to weight the width, height loss for small boxes.

    To avoid dividing by zero when going through the derivative formula of sqrt,
    Adds the eps value to the sqrt parameter.

    Derivative of sqrt : 1 / (2 * sqrt(x))
    """
    obj_true = y_true[:, :, :, 0]
    eps = tf.keras.backend.epsilon()

    xy_true = tf.sqrt(y_true[:, :, :, 1:3] + eps)
    xy_pred = tf.sqrt(y_pred[:, :, :, 1:3] + eps)
    xy_loss = __abs_log_loss(xy_true, xy_pred)
    xy_loss = tf.reduce_sum(xy_loss, axis=-1) * obj_true
    xy_loss = tf.reduce_mean(xy_loss, axis=0)
    xy_loss = tf.reduce_sum(xy_loss)

    wh_true = tf.sqrt(y_true[:, :, :, 3:5] + eps)
    wh_pred = tf.sqrt(y_pred[:, :, :, 3:5] + eps)
    wh_loss = __abs_log_loss(wh_true, wh_pred)
    wh_loss = tf.reduce_sum(wh_loss, axis=-1) * obj_true
    wh_loss = tf.reduce_mean(wh_loss, axis=0)
    wh_loss = tf.reduce_sum(wh_loss)
    bbox_loss = xy_loss + wh_loss
    return bbox_loss * 5.0


# def __constant_wh_loss(y_true, y_pred):
#     w_loss = __abs_log_loss(tf.constant(g_w_mean), y_pred[:, :, :, 3])
#     w_loss = tf.reduce_mean(w_loss, axis=0)
#     w_loss = tf.reduce_sum(w_loss)
#
#     h_loss = __abs_log_loss(tf.constant(g_h_mean), y_pred[:, :, :, 4])
#     h_loss = tf.reduce_mean(h_loss, axis=0)
#     h_loss = tf.reduce_sum(h_loss)
#     return w_loss + h_loss


# def __zero_classification_loss(y_true, y_pred):
#     classification_loss = __abs_log_loss(tf.constant(0.02), y_pred[:, :, :, 5:])
#     classification_loss = tf.reduce_sum(classification_loss, axis=-1)
#     classification_loss = tf.reduce_mean(classification_loss, axis=0)
#     classification_loss = tf.reduce_sum(classification_loss)
#     return classification_loss


def __classification_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    smooth_class_true = __smooth(y_true[:, :, :, 5:], alpha=0.02)
    classification_loss = __abs_log_loss(smooth_class_true, y_pred[:, :, :, 5:])
    classification_loss = tf.reduce_sum(classification_loss, axis=-1) * obj_true
    classification_loss = tf.reduce_mean(classification_loss, axis=0)
    classification_loss = tf.reduce_sum(classification_loss)
    return classification_loss


def confidence_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred)


def confidence_with_bbox_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred)


def burn_in_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    # return __zero_confidence_loss(y_true, y_pred) + __constant_wh_loss(y_true, y_pred) + __zero_classification_loss(y_true, y_pred)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred) + __classification_loss(y_true, y_pred)


def yolo_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred) + __classification_loss(y_true, y_pred)
