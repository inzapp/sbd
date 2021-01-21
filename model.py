import os

import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, output_channel):
        self.input_shape = input_shape
        self.output_channel = output_channel

    def build(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self._conv_block(16, 3, input_layer, True)
        x = self._conv_block(32, 3, x, True)
        x = self._conv_block(64, 3, x, True)
        x = self._conv_block(128, 3, x)
        x = self._conv_block(128, 3, x)
        x = self._point_wise_conv(self.output_channel, x)
        return tf.keras.models.Model(input_layer, x)

    @staticmethod
    def _conv_block(filters, kernel_size, x, max_pool=False):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if max_pool:
            x = tf.keras.layers.MaxPool2D()(x)
        return x

    @staticmethod
    def _point_wise_conv(filters, x):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            activation='sigmoid')(x)
