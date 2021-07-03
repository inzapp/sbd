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


def mean_absolute_logarithmic_error(y_true, y_pred):
    return tf.reduce_mean(-tf.math.log(1.0 + 1e-7 - tf.math.abs(y_pred - y_true)))


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
        return tf.losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])


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
        confidence_true = y_true[:, :, :, 0]
        x_loss = mean_absolute_logarithmic_error(y_true[:, :, :, 1], (y_pred[:, :, :, 1] * confidence_true))
        y_loss = mean_absolute_logarithmic_error(y_true[:, :, :, 2], (y_pred[:, :, :, 2] * confidence_true))
        w_true = tf.sqrt(y_true[:, :, :, 3] + 1e-4)
        w_pred = tf.sqrt(y_pred[:, :, :, 3] + 1e-4)
        w_loss = mean_absolute_logarithmic_error(w_true, (w_pred * confidence_true))
        h_true = tf.sqrt(y_true[:, :, :, 4] + 1e-4)
        h_pred = tf.sqrt(y_pred[:, :, :, 4] + 1e-4)
        h_loss = mean_absolute_logarithmic_error(h_true, (h_pred * confidence_true))
        bbox_loss = x_loss + y_loss + w_loss + h_loss
        #return confidence_loss + (bbox_loss * self.coord)
        return confidence_loss + bbox_loss


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(YoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        confidence_bbox_loss = ConfidenceWithBoundingBoxLoss()(y_true, y_pred)
        y_trues = tf.repeat(y_true[:, :, :, 0][..., tf.newaxis], self.num_classes, axis=-1)
        classification_loss = tf.losses.binary_crossentropy(y_true[:, :, :, 5:], y_pred[:, :, :, 5:] * y_trues, label_smoothing=0.1)
        return confidence_bbox_loss + classification_loss
