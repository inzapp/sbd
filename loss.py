"""
Authors : inzapp
Github url : https://github.com/inzapp/yolo

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


class AdjustConfidenceLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(AdjustConfidenceLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_sum(tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0]))


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, coord=5.0):
        """
        :param coord: coord value of bounding box loss. 5.0 is recommendation value in yolo paper.
        """
        self.coord = coord
        super(YoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        """
        SSE at all confidence channel
        no used lambda no_obj factor in here
        """
        confidence_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0]))

        """
        SSE at x, y regression loss
        """
        x_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 1] - (y_pred[:, :, :, 1] * y_true[:, :, :, 0])))
        y_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 2] - (y_pred[:, :, :, 2] * y_true[:, :, :, 0])))

        """
        SSE (sqrt(obj(x))) at width and height regression loss
        to avoid dividing by zero when going through the derivative formula of sqrt
        derivative of sqrt : 1 / 2 * sqrt(x)
        """
        w_true = tf.sqrt(y_true[:, :, :, 3] + 1e-4)
        w_pred = tf.sqrt(y_pred[:, :, :, 3] + 1e-4)
        w_loss = tf.reduce_sum(tf.square(w_true - (w_pred * y_true[:, :, :, 0])))
        h_true = tf.sqrt(y_true[:, :, :, 4] + 1e-4)
        h_pred = tf.sqrt(y_pred[:, :, :, 4] + 1e-4)
        h_loss = tf.reduce_sum(tf.square(h_true - (h_pred * y_true[:, :, :, 0])))
        bbox_loss = x_loss + y_loss + w_loss + h_loss

        """
        SSE at all classification loss
        """
        classification_loss = tf.reduce_sum(tf.reduce_sum(tf.square(y_true[:, :, :, 5:] - y_pred[:, :, :, 5:]), axis=-1) * y_true[:, :, :, 0])
        return confidence_loss + (bbox_loss * self.coord) + classification_loss
