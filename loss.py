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
        obj_false = (tf.ones(shape=tf.shape(obj_true), dtype=tf.dtypes.float32)) - obj_true

        # obj_confidence_loss = tf.reduce_sum(tf.square(smooth(obj_true, true_only=True) - (obj_pred * obj_true)))  #  conf sse
        obj_confidence_loss = tf.reduce_sum(tf.square((obj_pred * obj_true) - (self.iou(y_true, y_pred) * obj_true)))  # iou loss

        no_obj_confidence_loss = tf.reduce_sum(tf.square((obj_pred * obj_false) - tf.zeros(shape=tf.shape(obj_false), dtype=tf.dtypes.float32)))
        return obj_confidence_loss + (no_obj_confidence_loss * self.no_obj)

    @staticmethod
    def iou(y_true, y_pred):
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
        return intersection / (union + 1e-4)


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

        x_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 1] - (y_pred[:, :, :, 1] * confidence_true)))
        y_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 2] - (y_pred[:, :, :, 2] * confidence_true)))

        """
        SSE (sqrt(obj(x))) at width and height regression loss
        Sqrt was used to weight the width, height loss for small boxes.
        
        To avoid dividing by zero when going through the derivative formula of sqrt,
        Adds the eps value to the sqrt parameter.
        
        Derivative of sqrt : 1 / (2 * sqrt(x))
        """
        w_true = tf.sqrt(y_true[:, :, :, 3] + 1e-4)
        w_pred = tf.sqrt(y_pred[:, :, :, 3] + 1e-4)
        w_loss = tf.reduce_sum(tf.square(w_true - (w_pred * confidence_true)))
        h_true = tf.sqrt(y_true[:, :, :, 4] + 1e-4)
        h_pred = tf.sqrt(y_pred[:, :, :, 4] + 1e-4)
        h_loss = tf.reduce_sum(tf.square(h_true - (h_pred * confidence_true)))
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
        classification_loss = tf.reduce_sum(tf.reduce_sum(tf.square(smooth(y_true[:, :, :, 5:]) - y_pred[:, :, :, 5:]), axis=-1) * confidence_true)
        return confidence_bbox_loss + classification_loss
