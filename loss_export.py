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


def __confidence_loss(y_true, y_pred, ignore_threshold=ignore_threshold):
    obj_true = y_true[:, :, :, 0]
    obj_pred = y_pred[:, :, :, 0]
    loss = K.square(obj_true - obj_pred) * tf.where(obj_true * obj_pred > ignore_threshold, 0.0, 1.0)
    loss = K.mean(loss, axis=0)
    loss = K.sum(loss)
    return loss


def __bbox_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(K.sum(obj_true))
    if K.equal(obj_count, K.constant(0.0)):
        return 0.0

    xy_true = y_true[:, :, :, 1:3]
    xy_pred = y_pred[:, :, :, 1:3]
    xy_loss = K.square(xy_true - xy_pred)
    xy_loss = K.sum(xy_loss, axis=-1) * obj_true
    xy_loss = K.mean(xy_loss, axis=0)
    xy_loss = K.sum(xy_loss)

    eps = K.epsilon()
    wh_true = K.sqrt(y_true[:, :, :, 3:5] + eps)
    wh_pred = K.sqrt(y_pred[:, :, :, 3:5] + eps)
    wh_loss = K.square(wh_true - wh_pred) * obj_true
    wh_loss = K.sum(wh_loss, axis=-1)
    wh_loss = K.mean(wh_loss, axis=0)
    wh_loss = K.sum(wh_loss)
    return (xy_loss + wh_loss) * 5.0


def __classification_loss(y_true, y_pred):
    obj_true = y_true[:, :, :, 0]
    obj_count = K.cast_to_floatx(K.sum(obj_true))
    if K.equal(obj_count, K.constant(0.0)):
        return 0.0

    class_true = y_true[:, :, :, 5:]
    class_pred = y_pred[:, :, :, 5:]
    loss = K.square(class_true - class_pred)
    loss = K.sum(loss, axis=-1) * obj_true
    loss = K.mean(loss, axis=0)
    loss = K.sum(loss)
    return loss


def confidence_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred)


def confidence_with_bbox_loss(y_true, y_pred):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred) + __bbox_loss(y_true, y_pred)


def yolo_loss(y_true, y_pred, ignore_threshold=0.8):
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return __confidence_loss(y_true, y_pred, ignore_threshold=ignore_threshold) + __bbox_loss(y_true, y_pred) + __classification_loss(y_true, y_pred)
