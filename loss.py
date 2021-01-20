import tensorflow as tf


def loss(y_true, y_pred):
    from tensorflow.python.framework.ops import convert_to_tensor_v2
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    confidence_loss = tf.losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    x_loss = -tf.math.log(1.0 + 1e-7 - tf.reduce_mean(tf.abs(y_true[:, :, :, 1] - y_pred[:, :, :, 1]) * y_true[:, :, :, 0]))
    y_loss = -tf.math.log(1.0 + 1e-7 - tf.reduce_mean(tf.abs(y_true[:, :, :, 2] - y_pred[:, :, :, 2]) * y_true[:, :, :, 0]))
    w_true = tf.sqrt(y_true[:, :, :, 3] + 1e-4)
    w_pred = tf.sqrt(y_pred[:, :, :, 3] + 1e-4)
    w_loss = -tf.math.log(1.0 + 1e-7 - tf.reduce_mean(tf.abs(w_true - w_pred) * y_true[:, :, :, 0]))
    h_true = tf.sqrt(y_true[:, :, :, 4] + 1e-4)
    h_pred = tf.sqrt(y_pred[:, :, :, 4] + 1e-4)
    h_loss = -tf.math.log(1.0 + 1e-7 - tf.reduce_mean(tf.abs(h_true - h_pred) * y_true[:, :, :, 0]))
    bbox_loss = x_loss + y_loss + w_loss + h_loss
    classification_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(y_true[:, :, :, 5:] - y_pred[:, :, :, 5:]), axis=-1) * y_true[:, :, :, 0])
    classification_loss = -tf.math.log(1.0 + 1e-7 - classification_loss)

    # need to test below is valid
    # classification_losses = tf.constant(0.0)
    # for i in range(5, len(y_true)):
    #     cur_class_loss = tf.reduce_mean(tf.abs(y_true[:, :, :, i] - y_pred[:, :, :, i]) * y_true[:, :, :, 0])
    #     classification_losses += -tf.math.log(1.0 + 1e-7 - cur_class_loss)
    return confidence_loss + bbox_loss + classification_loss
