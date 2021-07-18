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


def smooth(y_true, alpha=0.1, true_only=False):
    smooth_true = tf.clip_by_value(y_true - alpha, 0.0, 1.0)
    if true_only:
        return smooth_true
    smooth_false = tf.clip_by_value(((y_true + alpha) * -1.0 + alpha * 2.0), 0.0, 1.0)
    return smooth_true + smooth_false


class ConfidenceLoss(tf.keras.losses.Loss):
    """
    This loss function is used to reduce the loss of the confidence channel with some epochs before training begins.
    """

    def __init__(self, no_obj=0.5):
        """
        :param no_obj: value for reduce gradient of confidence channel where no object. 0.5 is recommendation value in yolo paper.
        """
        self.no_obj = no_obj
        super(ConfidenceLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        obj_true = y_true[:, :, :, 0]
        obj_pred = y_pred[:, :, :, 0]
        obj_false = tf.ones(shape=tf.shape(obj_true), dtype=tf.dtypes.float32) - obj_true
        obj_confidence_loss = tf.reduce_mean(tf.square(smooth(obj_true, true_only=True) - (obj_pred * obj_true)))
        no_obj_confidence_loss = tf.reduce_mean(tf.square((obj_pred * obj_false) - tf.zeros(shape=tf.shape(obj_true), dtype=tf.dtypes.float32)))
        return obj_confidence_loss + (no_obj_confidence_loss * self.no_obj)


class ConfidenceWithBoundingBoxLoss(tf.keras.losses.Loss):
    """
    This loss function is used to reduce the loss of the confidence and bounding box channel with some epochs before training begins.
    """

    def __init__(self, coord=5.0):
        """
        :param coord: coord value of bounding box loss. 5.0 is recommendation value in yolo paper.
        """
        self.coord = coord
        super(ConfidenceWithBoundingBoxLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        confidence_loss = ConfidenceLoss()(y_true, y_pred)
        """
        SSE at x, y regression
        """
        confidence_true = y_true[:, :, :, 0]

        x_loss = tf.reduce_mean(tf.square(y_true[:, :, :, 1] - (y_pred[:, :, :, 1] * confidence_true)))
        y_loss = tf.reduce_mean(tf.square(y_true[:, :, :, 2] - (y_pred[:, :, :, 2] * confidence_true)))

        """
        SSE (sqrt(obj(x))) at width and height regression loss
        Sqrt was used to weight the width, height loss for small boxes.
        
        To avoid dividing by zero when going through the derivative formula of sqrt,
        Adds the eps value to the sqrt parameter.
        
        Derivative of sqrt : 1 / (2 * sqrt(x))
        """
        w_true = tf.sqrt(y_true[:, :, :, 3] + 1e-4)
        w_pred = tf.sqrt(y_pred[:, :, :, 3] + 1e-4)
        w_loss = tf.reduce_mean(tf.square(w_true - (w_pred * confidence_true)))
        h_true = tf.sqrt(y_true[:, :, :, 4] + 1e-4)
        h_pred = tf.sqrt(y_pred[:, :, :, 4] + 1e-4)
        h_loss = tf.reduce_mean(tf.square(h_true - (h_pred * confidence_true)))
        bbox_loss = x_loss + y_loss + w_loss + h_loss
        return confidence_loss + (bbox_loss * self.coord)


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        confidence_bbox_loss = ConfidenceWithBoundingBoxLoss()(y_true, y_pred)

        """
        SSE at all classification
        """
        confidence_true = y_true[:, :, :, 0]
        classification_loss = tf.reduce_mean(tf.reduce_mean(tf.square(smooth(y_true[:, :, :, 5:]) - y_pred[:, :, :, 5:]), axis=-1) * confidence_true)
        return confidence_bbox_loss + classification_loss
