import os

import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, output_channel):
        self.__input_shape = input_shape
        self.__output_channel = output_channel

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def build(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(16, 3, input_layer, True)
        x = self.__conv_block(32, 3, x, True)
        x = self.__conv_block(64, 3, x, True)
        x = self.__conv_block(128, 3, x, True)
        x = self.__conv_block(256, 3, x)
        x = self.__point_wise_conv(self.__output_channel, x)
        return tf.keras.models.Model(input_layer, x)

    @staticmethod
    def __conv_block(filters, kernel_size, x, max_pool=False):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer='he_uniform',
            padding='same',
            activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if max_pool:
            x = tf.keras.layers.MaxPool2D()(x)
        return x

    @staticmethod
    def __point_wise_conv(filters, x):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            activation='sigmoid')(x)
