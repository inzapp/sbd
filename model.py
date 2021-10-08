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


class Between(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.keras.backend.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}


class Model:
    def __init__(self, input_shape, output_channel, decay):
        self.__input_shape = input_shape
        self.__output_channel = output_channel
        self.__decay = decay

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def build(self):
        # return self.__vgg_16()
        # return self.__darknet_53()
        # return self.__lp_detection_sbd()
        # return self.__person_detail()
        # return self.__200m_big()
        # return self.__64_64_crop()
        # return self.__tiny_yolo_v3_no_upscale()
        # return self.__tiny_yolo_v4_no_upscale()
        # return self.__loon()
        return self.__loon_csp()

    def __200m_big(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(16, 3, input_layer, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(32, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(32, 3, x, bn=False)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=True)
        y1 = self.__detection_layer(x, 'output_1')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=True)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(512, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(512, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(512, 3, x, bn=True)
        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __64_64_crop(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(16, 3, input_layer, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(32, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(32, 3, x, bn=False)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=True)
        y1 = self.__detection_layer(x, 'output_1')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=True)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=True)
        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __tiny_yolo_v3_no_upscale(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(16, 3, input_layer, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(32, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(64, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(256, 3, x, bn=True)
        x = self.__conv_block(256, 1, x, bn=True)
        y1 = self.__detection_layer(x, 'output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(1024, 3, x, bn=True)
        x = self.__conv_block(256, 1, x, bn=True)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(256, 1, x, bn=True)
        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __tiny_yolo_v4_no_upscale(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(32, 3, input_layer, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(64, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__conv_block(32, 3, x, bn=False)
        x = self.__conv_block(32, 3, x, bn=False)
        x = self.__conv_block(64, 1, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__conv_block(128, 1, x, bn=True)
        y1 = self.__detection_layer(x, 'output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(256, 3, x, bn=False)
        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__conv_block(256, 1, x, bn=True)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x, bn=False)
        x = self.__conv_block(256, 1, x, bn=False)
        x = self.__conv_block(512, 3, x, bn=True)
        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __loon(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(8, 3, input_layer, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(16, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(32, 3, x, bn=True)
        x = self.__avg_max_pool(x)
        y1 = self.__detection_layer(x, 'output_1')

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(64, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __loon_csp(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(8, 3, input_layer, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__csp_block(16, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__csp_block(32, 3, x, bn=True)
        x = self.__avg_max_pool(x)
        y1 = self.__detection_layer(x, 'output_1')

        x = self.__drop_filter(x, 0.0625)
        x = self.__csp_block(64, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__csp_block(128, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __person_detail(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(16, 3, input_layer, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__conv_block(32, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__conv_block(64, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__conv_block(128, 3, x, bn=True)
        y1 = self.__detection_layer(x, 'output_1')
        x = self.__avg_max_pool(x)

        x = self.__conv_block(256, 3, x, bn=True)
        x = self.__conv_block(256, 3, x, bn=True)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__avg_max_pool(x)

        x = self.__conv_block(256, 3, x, bn=True)
        x = self.__conv_block(256, 3, x, bn=True)
        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __lcd(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(16, 3, input_layer, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(32, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(64, 3, x, bn=True)
        x = self.__max_pool(x)

        y1 = self.__detection_layer(x, 'output_1')
        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__max_pool(x)

        y2 = self.__detection_layer(x, 'output_2')
        x = self.__conv_block(256, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x, bn=True)
        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __lp_detection_sbd(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(8, 3, input_layer, bn=True, activation_first=True)
        x = self.__max_pool(x)

        x = self.__conv_block(16, 3, x, bn=True, activation_first=True)
        x = self.__max_pool(x)

        x = self.__conv_block(32, 3, x, bn=True, activation_first=True)
        x = self.__max_pool(x)

        x = self.__conv_block(64, 3, x, bn=True, activation_first=True)
        x = self.__max_pool(x)

        x = self.__conv_block(64, 3, x, bn=True, activation_first=True)
        y1 = self.__detection_layer(x, 'output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(128, 3, x, bn=True, activation_first=True)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(128, 3, x, bn=True, activation_first=True)
        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __vgg_16(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(64, 3, input_layer, bn=True)
        x = self.__conv_block(64, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(256, 3, x, bn=True)
        x = self.__conv_block(256, 3, x, bn=True)
        x = self.__conv_block(256, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(512, 3, x, bn=True)
        y1 = self.__detection_layer(x, name='output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(512, 3, x, bn=True)
        y2 = self.__detection_layer(x, name='output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(512, 3, x, bn=True)
        y3 = self.__detection_layer(x, name='output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __darknet_19(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(32, 3, input_layer, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(64, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__conv_block(64, 1, x, bn=True)
        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__max_pool(x)

        x = self.__conv_block(256, 3, x, bn=True)
        x = self.__conv_block(128, 1, x, bn=True)
        x = self.__conv_block(256, 3, x, bn=True)
        y1 = self.__detection_layer(x, 'output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(256, 1, x, bn=True)
        x = self.__conv_block(512, 3, x, bn=True)
        x = self.__conv_block(256, 1, x, bn=True)
        x = self.__conv_block(512, 3, x, bn=True)
        y2 = self.__detection_layer(x, 'output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(1024, 3, x, bn=True)
        x = self.__conv_block(512, 1, x, bn=True)
        x = self.__conv_block(1024, 3, x, bn=True)
        x = self.__conv_block(512, 1, x, bn=True)
        x = self.__conv_block(1024, 3, x, bn=True)
        y3 = self.__detection_layer(x, 'output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __darknet_53(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_blocks(1, 32, 3, input_layer)
        x = self.__conv_blocks(1, 64, 3, x)
        x = self.__max_pool(x)
        skip_connection = x

        x = self.__conv_blocks(1, 32, 1, x)
        x = self.__conv_blocks(1, 64, 3, x)
        x = tf.keras.layers.Add()([skip_connection, x])
        x = self.__conv_blocks(1, 128, 3, x)
        x = self.__max_pool(x)
        skip_connection = x

        for _ in range(2):
            x = self.__conv_blocks(1, 64, 1, x)
            x = self.__conv_blocks(1, 128, 3, x)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 256, 3, x)
        x = self.__max_pool(x)
        y1 = self.__detection_layer(x, name='output_1')
        skip_connection = x

        for _ in range(8):
            x = self.__conv_blocks(1, 128, 1, x)
            x = self.__conv_blocks(1, 256, 3, x)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 512, 3, x)
        x = self.__max_pool(x)
        y2 = self.__detection_layer(x, name='output_2')
        skip_connection = x

        for _ in range(4):
            x = self.__conv_blocks(1, 256, 1, x)
            x = self.__conv_blocks(1, 512, 3, x)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 1024, 3, x)
        x = self.__max_pool(x)
        skip_connection = x

        for _ in range(4):
            x = self.__conv_blocks(1, 512, 1, x)
            x = self.__conv_blocks(1, 1024, 3, x)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        y3 = self.__detection_layer(x, name='output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __conv_blocks(self, n_convolutions, filters, kernel_size, x, activation_first=False, bn=True):
        for _ in range(n_convolutions):
            x = self.__conv_block(filters, kernel_size, x, activation_first=activation_first, bn=bn)
        return x

    def __csp_block(self, filters, kernel_size, x, activation_first=False, bn=True):
        x_0 = self.__conv_block(filters / 2, 1, x, activation_first=activation_first, bn=bn)
        x_1 = self.__conv_block(filters / 2, 1, x, activation_first=activation_first, bn=bn)
        x_1 = self.__conv_block(filters / 2, kernel_size, x_1, activation_first=activation_first, bn=bn)
        x_1_0 = self.__conv_block(filters / 4, 1, x_1, activation_first=activation_first, bn=bn)
        x_1_1 = self.__conv_block(filters / 4, 1, x_1, activation_first=activation_first, bn=bn)
        x_1_1 = self.__conv_block(filters / 4, kernel_size, x_1_1, activation_first=activation_first, bn=bn)
        x_1 = tf.keras.layers.Concatenate()([x_1_0, x_1_1])
        x = tf.keras.layers.Concatenate()([x_0, x_1])
        return x

    def __conv_block(self, filters, kernel_size, x, activation_first=False, bn=True):
        x = self.__conv(x, filters, kernel_size, use_bias=False if bn else True)
        if activation_first:
            x = self.__relu(x)
            if bn:
                x = self.__bn(x)
        else:
            if bn:
                x = self.__bn(x)
            x = self.__relu(x)
        return x

    def __conv(self, x, filters, kernel_size, use_bias=True):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.zeros(),
            padding='same',
            use_bias=use_bias,
            # bias_constraint=tf.keras.constraints.min_max_norm(min_value=-1.0, max_value=1.0),
            # bias_constraint=tf.keras.constraints.max_norm(3),
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.__decay) if self.__decay > 0.0 else None)(x)

    def __detection_layer(self, x, name='output'):
        return tf.keras.layers.Conv2D(
            filters=self.__output_channel,
            kernel_size=1,
            activation='sigmoid',
            name=name)(x)

    @staticmethod
    def standardization(w):
        w_mean = tf.reduce_mean(w, axis=[0, 1, 2], keepdims=True)
        w = w - w_mean
        w_std = tf.keras.backend.std(w, axis=[0, 1, 2], keepdims=True)
        w = w / (w_std + 1e-5)
        return w

    @staticmethod
    def __max_pool(x):
        return tf.keras.layers.MaxPool2D()(x)

    @staticmethod
    def __avg_max_pool(x):
        ap = tf.keras.layers.AvgPool2D()(x)
        mp = tf.keras.layers.MaxPool2D()(x)
        return tf.keras.layers.Add()([ap, mp])

    @staticmethod
    def __drop_filter(x, rate):
        return tf.keras.layers.SpatialDropout2D(rate)(x)

    @staticmethod
    def __dropout(x, rate):
        return tf.keras.layers.Dropout(rate)(x)

    @staticmethod
    def __bn(x):
        return tf.keras.layers.BatchNormalization(beta_initializer=tf.keras.initializers.zeros(), fused=True)(x)

    @staticmethod
    def __relu(x):
        return tf.keras.layers.ReLU()(x)
