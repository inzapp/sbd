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
import os

import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, output_channel, decay):
        self.__input_shape = input_shape
        self.__output_channel = output_channel
        self.__decay = decay

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def build(self):
        # return self.__vgg_19()
        # return self.__darknet_53()
        return self.__lp_detection_sbd()
        # return self.__person_detail()
        # return self.__200m()
        # return self.__200m_crop()
        # return self.__covid()
        # return self.__loon()

    def __200m_crop(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_blocks(1, 16, 3, input_layer, avg_max_pool=True)
        x = self.__conv_blocks(2, 32, 3, x, avg_max_pool=True)
        x = self.__conv_blocks(2, 64, 3, x, avg_max_pool=True)

        x = self.__conv_blocks(2, 128, 3, x, activation_first=True)
        y1 = self.__point_wise_conv(self.__output_channel, x, 'output_1')
        x = self.__avg_max_pool(x)

        x = self.__conv_blocks(2, 128, 3, x, activation_first=True)
        y2 = self.__point_wise_conv(self.__output_channel, x, 'output_2')
        x = self.__avg_max_pool(x)

        x = self.__conv_blocks(2, 128, 3, x, activation_first=True)
        y3 = self.__point_wise_conv(self.__output_channel, x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __200m(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_blocks(1, 8, 3, input_layer, avg_max_pool=True)
        x = self.__conv_blocks(2, 16, 3, x, avg_max_pool=True)
        x = self.__conv_blocks(2, 32, 3, x, avg_max_pool=True)
        x = self.__conv_blocks(2, 64, 3, x, avg_max_pool=True)

        # x = self.__dropout(x, 0.1)
        x = self.__conv_blocks(2, 128, 3, x, activation_first=True)
        y1 = self.__point_wise_conv(self.__output_channel, x, 'output_1')
        x = self.__avg_max_pool(x)

        # x = self.__dropout(x, 0.2)
        x = self.__conv_blocks(2, 256, 3, x, activation_first=True)
        y2 = self.__point_wise_conv(self.__output_channel, x, 'output_2')
        x = self.__avg_max_pool(x)

        # x = self.__dropout(x, 0.3)
        x = self.__conv_blocks(2, 256, 3, x, activation_first=True)
        y3 = self.__point_wise_conv(self.__output_channel, x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __loon(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(8, 3, input_layer, )
        x = self.__conv_block(16, 3, x, avg_max_pool=True)
        x = self.__conv_block(32, 3, x, avg_max_pool=True)
        y1 = self.__point_wise_conv(self.__output_channel, x, 'output_1')
        x = self.__conv_block(64, 3, x, avg_max_pool=True)
        y2 = self.__point_wise_conv(self.__output_channel, x, 'output_2')
        x = self.__conv_block(128, 3, x, avg_max_pool=True)
        y3 = self.__point_wise_conv(self.__output_channel, x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    # input_shape=(128, 64, 1) or input_shape=(192, 96, 1)
    def __person_detail(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(16, 3, input_layer, avg_max_pool=True)
        x = self.__conv_block(32, 3, x, avg_max_pool=True)
        x = self.__conv_block(64, 3, x, avg_max_pool=True)

        x = self.__conv_blocks(2, 128, 3, x)
        y1 = self.__point_wise_conv(self.__output_channel, x, 'output_1')
        x = self.__avg_max_pool(x)

        x = self.__conv_blocks(2, 128, 3, x)
        y2 = self.__point_wise_conv(self.__output_channel, x, 'output_2')
        x = self.__avg_max_pool(x)

        x = self.__conv_blocks(2, 128, 3, x)
        y3 = self.__point_wise_conv(self.__output_channel, x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __covid(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_blocks(2, 4, 3, input_layer, max_pool=True)
        x = self.__conv_blocks(3, 8, 3, x, max_pool=True)

        x = self.__conv_blocks(3, 16, 3, x, max_pool=True)
        x = self.__conv_blocks(3, 32, 3, x)
        y1 = self.__point_wise_conv(self.__output_channel, x, name='output_1')
        x = self.__max_pool(x)

        x = self.__conv_blocks(3, 32, 3, x)
        y2 = self.__point_wise_conv(self.__output_channel, x, name='output_2')
        x = self.__max_pool(x)

        x = self.__conv_blocks(3, 32, 3, x)
        y3 = self.__point_wise_conv(self.__output_channel, x, name='output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    # input_shape=(96, 192, 1) or input_shape=(144, 288, 1)
    def __lcd(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(16, 3, input_layer, max_pool=True)
        x = self.__conv_block(32, 3, x, max_pool=True)
        x = self.__conv_block(64, 3, x, max_pool=True)
        y1 = self.__point_wise_conv(self.__output_channel, x, 'output_1')
        x = self.__conv_block(128, 3, x, max_pool=True)
        y2 = self.__point_wise_conv(self.__output_channel, x, 'output_2')
        x = self.__conv_block(256, 3, x, max_pool=True)
        x = self.__conv_block(512, 3, x)
        y3 = self.__point_wise_conv(self.__output_channel, x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    # input_shape=(368, 640, 1)
    def __lp_detection_sbd(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(8, 3, input_layer, activation_first=True, max_pool=True)
        x = self.__conv_block(16, 3, x, activation_first=True, max_pool=True)
        x = self.__conv_block(32, 3, x, activation_first=True, max_pool=True)
        x = self.__conv_block(64, 3, x, activation_first=True, max_pool=True)
        x = self.__conv_block(64, 3, x, activation_first=True)
        y1 = self.__point_wise_conv(self.__output_channel, x, 'output_1')
        x = self.__max_pool(x)
        x = self.__conv_block(128, 3, x, activation_first=True)
        y2 = self.__point_wise_conv(self.__output_channel, x, 'output_2')
        x = self.__max_pool(x)
        x = self.__conv_block(128, 3, x, activation_first=True)
        y3 = self.__point_wise_conv(self.__output_channel, x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __darknet_19(self):
        if self.__input_shape[0] < 224 and self.__input_shape[1] < 224:
            print('[ERROR] minimum input size of darknet 19 is (224, 224). consider using smaller networks to train images of smaller sizes.')
            exit(-1)

        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(32, 3, input_layer, max_pool=True)
        x = self.__conv_block(64, 3, x, max_pool=True)

        x = self.__conv_block(128, 3, x)
        x = self.__conv_block(64, 1, x)
        x = self.__conv_block(128, 3, x, max_pool=True)

        x = self.__conv_block(256, 3, x)
        x = self.__conv_block(128, 1, x)
        x = self.__conv_block(256, 3, x)
        y1 = self.__point_wise_conv(self.__output_channel, x, 'output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x)
        x = self.__conv_block(256, 1, x)
        x = self.__conv_block(512, 3, x)
        x = self.__conv_block(256, 1, x)
        x = self.__conv_block(512, 3, x)
        y2 = self.__point_wise_conv(self.__output_channel, x, 'output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(1024, 3, x)
        x = self.__conv_block(512, 1, x)
        x = self.__conv_block(1024, 3, x)
        x = self.__conv_block(512, 1, x)
        x = self.__conv_block(1024, 3, x)
        y3 = self.__point_wise_conv(self.__output_channel, x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __darknet_53(self):
        if self.__input_shape[0] < 224 and self.__input_shape[1] < 224:
            print('[ERROR] minimum input size of darknet 53 is (224, 224). consider using smaller networks to train images of smaller sizes.')
            exit(-1)

        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_blocks(1, 32, 3, input_layer)
        x = self.__conv_blocks(1, 64, 3, x, max_pool=True)
        skip_connection = x

        x = self.__conv_blocks(1, 32, 1, x)
        x = self.__conv_blocks(1, 64, 3, x)
        x = tf.keras.layers.Add()([skip_connection, x])
        x = self.__conv_blocks(1, 128, 3, x, max_pool=True)
        skip_connection = x

        for _ in range(2):
            x = self.__conv_blocks(1, 64, 1, x)
            x = self.__conv_blocks(1, 128, 3, x)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 256, 3, x, max_pool=True)
        y1 = self.__point_wise_conv(self.__output_channel, x, name='output_1')
        skip_connection = x

        for _ in range(8):
            x = self.__conv_blocks(1, 128, 1, x)
            x = self.__conv_blocks(1, 256, 3, x)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 512, 3, x, max_pool=True)
        y2 = self.__point_wise_conv(self.__output_channel, x, name='output_2')
        skip_connection = x

        for _ in range(4):
            x = self.__conv_blocks(1, 256, 1, x)
            x = self.__conv_blocks(1, 512, 3, x)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 1024, 3, x, max_pool=True)
        skip_connection = x

        for _ in range(4):
            x = self.__conv_blocks(1, 512, 1, x)
            x = self.__conv_blocks(1, 1024, 3, x)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        y3 = self.__point_wise_conv(self.__output_channel, x, name='output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __vgg_19(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_blocks(2, 64, 3, input_layer, max_pool=True)
        x = self.__conv_blocks(3, 128, 3, x, max_pool=True)

        x = self.__conv_blocks(4, 256, 3, x)
        y1 = self.__point_wise_conv(self.__output_channel, x, name='output_1')
        x = self.__max_pool(x)

        x = self.__conv_blocks(4, 512, 3, x)
        y2 = self.__point_wise_conv(self.__output_channel, x, name='output_2')
        x = self.__max_pool(x)

        x = self.__conv_blocks(4, 512, 3, x)
        y3 = self.__point_wise_conv(self.__output_channel, x, name='output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __conv_blocks(self, n_convolutions, filters, kernel_size, x, max_pool=False, avg_max_pool=False, activation_first=False):
        for _ in range(n_convolutions):
            x = self.__conv_block(filters, kernel_size, x, activation_first=activation_first)
        if max_pool:
            x = self.__max_pool(x)
        elif avg_max_pool:
            x = self.__avg_max_pool(x)
        return x

    def __conv_block(self, filters, kernel_size, x, max_pool=False, avg_max_pool=False, activation_first=False):
        if activation_first:
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                kernel_initializer='he_uniform',
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.__decay))(x)
            x = tf.keras.layers.BatchNormalization()(x)
        else:
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                kernel_initializer='he_uniform',
                padding='same',
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.__decay))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        if max_pool:
            x = self.__max_pool(x)
        elif avg_max_pool:
            x = self.__avg_max_pool(x)
        return x

    @staticmethod
    def __max_pool(x):
        return tf.keras.layers.MaxPool2D()(x)

    @staticmethod
    def __avg_max_pool(x):
        ap = tf.keras.layers.AvgPool2D()(x)
        mp = tf.keras.layers.MaxPool2D()(x)
        return tf.keras.layers.Add()([ap, mp])

    @staticmethod
    def __dropout(x, rate):
        return tf.keras.layers.Dropout(rate)(x)

    @staticmethod
    def __point_wise_conv(filters, x, name='output'):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            activation='sigmoid',
            name=name)(x)
