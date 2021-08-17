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


def __confidence_loss(y_true, y_pred):
    no_obj = 0.5
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    obj_false = tf.ones(shape=tf.shape(obj_true), dtype=tf.dtypes.float32) - obj_true

    obj_confidence_loss = tf.keras.backend.binary_crossentropy(obj_true, obj_pred, from_logits=False) * obj_true
    obj_confidence_loss = tf.reduce_mean(obj_confidence_loss, axis=0)
    obj_confidence_loss = tf.reduce_sum(obj_confidence_loss)

    zeros = tf.zeros(shape=tf.shape(obj_true), dtype=tf.dtypes.float32)
    no_obj_confidence_loss = tf.keras.backend.binary_crossentropy(obj_pred, zeros, from_logits=False) * obj_false
    no_obj_confidence_loss = tf.reduce_mean(no_obj_confidence_loss, axis=0)
    no_obj_confidence_loss = tf.reduce_sum(no_obj_confidence_loss)
    return obj_confidence_loss + (no_obj_confidence_loss * no_obj)


def __bbox_loss(y_true, y_pred):
    """
    BCE (sqrt(obj(x))) at width and height regression loss
    Sqrt was used to weight the width, height loss for small boxes.

    To avoid dividing by zero when going through the derivative formula of sqrt,
    Adds the eps value to the sqrt parameter.

    Derivative of sqrt : 1 / (2 * sqrt(x))
    """
    obj_true = y_true[:, :, :, 0]

    xy_true = y_true[:, :, :, 1:3]
    xy_pred = y_pred[:, :, :, 1:3]
    xy_loss = tf.keras.backend.binary_crossentropy(xy_true, xy_pred, from_logits=False)
    xy_loss = tf.reduce_sum(xy_loss, axis=-1) * obj_true
    xy_loss = tf.reduce_mean(xy_loss, axis=0)
    xy_loss = tf.reduce_sum(xy_loss)

    eps = tf.keras.backend.epsilon()
    wh_true = tf.sqrt(y_true[:, :, :, 3:5] + eps)
    wh_pred = tf.sqrt(y_pred[:, :, :, 3:5] + eps)
    wh_loss = tf.keras.backend.binary_crossentropy(wh_true, wh_pred, from_logits=False)
    wh_loss = tf.reduce_sum(wh_loss, axis=-1) * obj_true
    wh_loss = tf.reduce_mean(wh_loss, axis=0)
    wh_loss = tf.reduce_sum(wh_loss)
    bbox_loss = xy_loss + wh_loss
    return bbox_loss


def __classification_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    classification_loss = tf.keras.backend.binary_crossentropy(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], from_logits=False)
    classification_loss = tf.reduce_sum(classification_loss, axis=-1) * obj_true
    classification_loss = tf.reduce_mean(classification_loss, axis=0)
    classification_loss = tf.reduce_sum(classification_loss)
    return classification_loss


def confidence_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.sigmoid(y_pred)
    return __confidence_loss(y_true, y_pred)


def confidence_with_bbox_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.sigmoid(y_pred)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred)


def yolo_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.sigmoid(y_pred)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred) + __classification_loss(y_true, y_pred)
