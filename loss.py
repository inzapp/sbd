import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor_v2


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, coord=5.0):
        self.coord = coord
        super(YoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        confidence_loss = tf.losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])
        x_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 1] - (y_pred[:, :, :, 1] * y_true[:, :, :, 0])))
        y_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 2] - (y_pred[:, :, :, 2] * y_true[:, :, :, 0])))
        w_true = tf.sqrt(y_true[:, :, :, 3] + 1e-4)
        w_pred = tf.sqrt(y_pred[:, :, :, 3] + 1e-4)
        w_loss = tf.reduce_sum(tf.square(w_true - (w_pred * y_true[:, :, :, 0])))
        h_true = tf.sqrt(y_true[:, :, :, 4] + 1e-4)
        h_pred = tf.sqrt(y_pred[:, :, :, 4] + 1e-4)
        h_loss = tf.reduce_sum(tf.square(h_true - (h_pred * y_true[:, :, :, 0])))
        bbox_loss = x_loss + y_loss + w_loss + h_loss
        classification_loss = tf.reduce_sum(tf.reduce_sum(tf.square(y_true[:, :, :, 5:] - y_pred[:, :, :, 5:]), axis=-1) * y_true[:, :, :, 0])
        return confidence_loss + (bbox_loss * self.coord) + classification_loss
