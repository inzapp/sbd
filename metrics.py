import tensorflow as tf


def precision(y_true, y_pred):
    tp = tf.reduce_sum(y_pred[:, :, :, 0] * y_true[:, :, :, 0])
    return tp / (tf.reduce_sum(y_pred[:, :, :, 0]) + 1e-5)


def recall(y_true, y_pred):
    tp = tf.reduce_sum(y_pred[:, :, :, 0] * y_true[:, :, :, 0])
    return tp / (tf.reduce_sum(y_true[:, :, :, 0]) + 1e-5)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (p * r * 2.0) / (p + r + 1e-5)
