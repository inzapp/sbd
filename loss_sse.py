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


def __confidence_loss(y_true, y_pred, ignore_threshold):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    loss = tf.square(obj_true - obj_pred)
    obj_loss = tf.reduce_sum(tf.reduce_mean(loss * obj_true, axis=0))
    background_loss = tf.reduce_sum(tf.reduce_mean(loss * (1.0 - obj_true), axis=0))
    return obj_loss + (background_loss * 0.5)


def __bbox_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(tf.reduce_sum(obj_true))
    if tf.equal(obj_count, tf.constant(0.0)):
        return 0.0

    xy_true = y_true[:, :, :, 1:3]
    xy_pred = y_pred[:, :, :, 1:3]
    xy_loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.square(xy_true - xy_pred), axis=-1) * obj_true, axis=0))

    eps = K.epsilon()
    wh_true = tf.sqrt(y_true[:, :, :, 3:5] + eps)
    wh_pred = tf.sqrt(y_pred[:, :, :, 3:5] + eps)
    wh_loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.square(wh_true - wh_pred), axis=-1) * obj_true, axis=0))
    return (xy_loss + wh_loss) * 5.0


def __classification_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(tf.reduce_sum(obj_true))
    if tf.equal(obj_count, tf.constant(0.0)):
        return 0.0

    class_true = y_true[:, :, :, 5:]
    class_pred = y_pred[:, :, :, 5:]
    loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.square(class_true - class_pred), axis=-1) * obj_true, axis=0))
    return loss


def confidence_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred)


def confidence_with_bbox_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred)


def yolo_loss(y_true, y_pred, ignore_threshold):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, ignore_threshold) + __bbox_loss(y_true, y_pred) + __classification_loss(y_true, y_pred)

