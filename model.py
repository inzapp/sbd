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
        # return self.__normal_model()
        # return self.__vgg_16()
        # return self.__darknet_53()
        # return self.__lp_detection_sbd()
        # return self.__lp_detection_sbd_csp()
        # return self.__person_detail()
        # return self.__person_part_detail()
        # return self.__lcd_csp()
        # return self.__200m_big()
        # return self.__tiny_yolo_v3_no_upscale()
        return self.__loon()
        # return self.__loon_csp()
        # return self.__loon_cross()
        # return self.__loon_cross_csp()

    def __normal_model(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 32, 3, bn=True, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 64, 3, bn=False, activation='swish')
        x = self.__conv_block(x, 64, 3, bn=True, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 128, 3, bn=False, activation='swish')
        x = self.__conv_block(x, 128, 3, bn=True, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 256, 3, bn=False, activation='swish')
        x = self.__conv_block(x, 256, 3, bn=True, activation='swish')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 512, 3, bn=False, activation='swish')
        x = self.__conv_block(x, 512, 3, bn=True, activation='swish')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 512, 3, bn=False, activation='swish')
        x = self.__conv_block(x, 512, 3, bn=True, activation='swish')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __200m_big(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 16, 3, bn=True, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 32, 3, bn=False, activation='relu')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 32, 3, bn=False, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 64, 3, bn=False, activation='relu')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 64, 3, bn=False, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 128, 3, bn=False, activation='relu')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 128, 3, bn=False, activation='relu')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 256, 3, bn=False, activation='relu')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 256, 3, bn=False, activation='relu')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 256, 3, bn=True, activation='relu')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 512, 3, bn=False, activation='relu')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 512, 3, bn=False, activation='relu')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __tiny_yolo_v3_no_upscale(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 16, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 32, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 64, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 256, 1, bn=True, activation='relu')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 1024, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 256, 1, bn=True, activation='relu')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 256, 1, bn=True, activation='relu')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __loon(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 8, 3, bn=True, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 16, 3, bn=True, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 32, 3, bn=True, activation='swish')
        x = self.__max_pool(x)
        y1 = self.__detection_layer(x, 'sbd_output_1')

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 64, 3, bn=True, activation='swish')
        x = self.__max_pool(x)
        y2 = self.__detection_layer(x, 'sbd_output_2')

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 128, 3, bn=True, activation='swish')
        x = self.__max_pool(x)
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __loon_csp(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 8, 3, bn=True, activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 16, 3, bn=True, activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__csp_block(x, 32, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='swish', inner_activation='swish')
        x = self.__avg_max_pool(x)
        y1 = self.__detection_layer(x, 'sbd_output_1')

        x = self.__drop_filter(x, 0.0625)
        x = self.__csp_block(x, 64, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='swish', inner_activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__csp_block(x, 128, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='swish', inner_activation='swish')
        x = self.__avg_max_pool(x)

        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __loon_cross(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__cross_conv_block(input_layer, 8, 3, bn=True, mode='add', activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 16, 3, bn=True, mode='add', activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 32, 3, bn=True, mode='add', activation='swish')
        x = self.__avg_max_pool(x)
        y1 = self.__detection_layer(x, 'sbd_output_1')

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 64, 3, bn=True, mode='add', activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__cross_conv_block(x, 128, 3, bn=True, mode='add', activation='swish')
        x = self.__avg_max_pool(x)

        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __loon_cross_csp(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__cross_conv_block(input_layer, 8, 3, bn=True, activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 16, 3, bn=True, activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_csp_block(x, 32, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, mode='add', activation='swish', inner_activation='swish')
        x = self.__avg_max_pool(x)
        y1 = self.__detection_layer(x, 'sbd_output_1')

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_csp_block(x, 64, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, mode='add', activation='swish', inner_activation='swish')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__cross_csp_block(x, 128, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, mode='add', activation='swish', inner_activation='swish')
        x = self.__avg_max_pool(x)

        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __person_detail(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 16, 3, bn=True, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 32, 3, bn=False, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 64, 3, bn=False, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 128, 3, bn=False, activation='swish')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 128, 3, bn=True, activation='swish')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 256, 3, bn=False, activation='swish')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 256, 3, bn=True, activation='swish')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 256, 3, bn=False, activation='swish')
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 256, 3, bn=True, activation='swish')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __person_part_detail(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__cross_conv_block(input_layer, 16, 3, bn=True, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 32, 3, bn=False, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 64, 3, bn=False, activation='swish')
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 128, 3, bn=False, activation='swish')
        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 128, 3, bn=True, activation='swish')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 256, 3, bn=False, activation='swish')
        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 256, 3, bn=True, activation='swish')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 256, 3, bn=False, activation='swish')
        x = self.__drop_filter(x, 0.0625)
        x = self.__cross_conv_block(x, 256, 3, bn=True, activation='swish')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __lcd(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 16, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 32, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 64, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __lcd_csp(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 16, 3, bn=True, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__conv_block(x, 32, 3, bn=False, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__conv_block(x, 64, 3, bn=False, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__csp_block(x, 128, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='relu', inner_activation='relu')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__avg_max_pool(x)

        x = self.__csp_block(x, 128, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='relu', inner_activation='relu')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__avg_max_pool(x)

        x = self.__csp_block(x, 256, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='relu', inner_activation='relu')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __lp_detection_sbd(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 8, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 16, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 32, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 64, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 64, 3, bn=True, activation='relu')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __lp_detection_sbd_csp(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 8, 3, bn=True, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 16, 3, bn=False, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(x, 32, 3, bn=False, activation='relu')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__csp_block(64, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='relu', inner_activation='relu')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__csp_block(128, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='relu', inner_activation='relu')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__csp_block(128, 3, first_depth_n_convs=1, second_depth_n_convs=2, bn=True, activation='relu', inner_activation='relu')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __vgg_16(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 64, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 64, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        y1 = self.__detection_layer(x, name='sbd_output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        y2 = self.__detection_layer(x, name='sbd_output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        y3 = self.__detection_layer(x, name='sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __darknet_19(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_block(input_layer, 32, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 64, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 64, 1, bn=True, activation='relu')
        x = self.__conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 128, 1, bn=True, activation='relu')
        x = self.__conv_block(x, 256, 3, bn=True, activation='relu')
        y1 = self.__detection_layer(x, 'sbd_output_1')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 256, 1, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 256, 1, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 3, bn=True, activation='relu')
        y2 = self.__detection_layer(x, 'sbd_output_2')
        x = self.__max_pool(x)

        x = self.__conv_block(x, 1024, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 1, bn=True, activation='relu')
        x = self.__conv_block(x, 1024, 3, bn=True, activation='relu')
        x = self.__conv_block(x, 512, 1, bn=True, activation='relu')
        x = self.__conv_block(x, 1024, 3, bn=True, activation='relu')
        y3 = self.__detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def __csp_block(self, x, filters, kernel_size, first_depth_n_convs=1, second_depth_n_convs=2, bn=False, activation='none', inner_activation='none'):
        half_filters = filters / 2
        x_0 = self.__conv_block(x, half_filters, 1, bn=False, activation='none')
        for i in range(first_depth_n_convs):
            if i == 0:
                x_1 = self.__conv_block(x, half_filters, 1, bn=False, activation=inner_activation)
            else:
                x_1 = self.__conv_block(x_1, half_filters, kernel_size, bn=False, activation=inner_activation)
        x_1_0 = self.__conv_block(x_1, half_filters, 1, bn=False, activation='none')
        for i in range(second_depth_n_convs):
            if i == 0:
                x_1_1 = self.__conv_block(x_1, half_filters, 1, bn=False, activation=inner_activation)
            else:
                x_1_1 = self.__conv_block(x_1_1, half_filters, kernel_size, bn=False, activation='none' if i == second_depth_n_convs - 1 else inner_activation)
        x_1 = tf.keras.layers.Concatenate()([x_1_0, x_1_1])
        x = tf.keras.layers.Concatenate()([x_0, x_1])
        if bn:
            x = self.__bn(x)
        x = self.__activation(x, activation=activation)
        return x

    def __conv_block(self, x, filters, kernel_size, bn=True, activation='none'):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.zeros(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.__decay) if self.__decay > 0.0 else None)(x)
        if bn:
            x = self.__bn(x)
        x = self.__activation(x, activation=activation)
        return x

    def __cross_csp_block(self, x, filters, kernel_size, first_depth_n_convs=1, second_depth_n_convs=2, bn=False, mode='add', activation='none', inner_activation='none'):
        half_filters = filters / 2
        x_0 = self.__cross_conv_block(x, half_filters, 1, bn=False, mode=mode, activation='none')
        for i in range(first_depth_n_convs):
            if i == 0:
                x_1 = self.__cross_conv_block(x, half_filters, 1, bn=False, mode=mode, activation=inner_activation)
            else:
                x_1 = self.__cross_conv_block(x_1, half_filters, kernel_size, bn=False, mode=mode, activation=inner_activation)
        x_1_0 = self.__cross_conv_block(x_1, half_filters, 1, bn=False, activation='none')
        for i in range(second_depth_n_convs):
            if i == 0:
                x_1_1 = self.__cross_conv_block(x_1, half_filters, 1, bn=False, mode=mode, activation=inner_activation)
            else:
                x_1_1 = self.__cross_conv_block(x_1_1, half_filters, kernel_size, bn=False, mode=mode, activation='none' if i == second_depth_n_convs - 1 else inner_activation)
        x_1 = tf.keras.layers.Concatenate()([x_1_0, x_1_1])
        x = tf.keras.layers.Concatenate()([x_0, x_1])
        if bn:
            x = self.__bn(x)
        x = self.__activation(x, activation=activation)
        return x

    def __cross_conv_block(self, x, filters, kernel_size, bn=True, mode='add', activation='none'):
        v_conv = tf.keras.layers.Conv2D(
            filters=filters if mode == 'add' else filters / 2,
            kernel_size=(1, kernel_size),
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.zeros(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.__decay) if self.__decay > 0.0 else None)(x)
        h_conv = tf.keras.layers.Conv2D(
            filters=filters if mode == 'add' else filters / 2,
            kernel_size=(kernel_size, 1),
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.zeros(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.__decay) if self.__decay > 0.0 else None)(x)
        x = tf.keras.layers.Add()([v_conv, h_conv]) if mode == 'add' else tf.keras.layers.Concatenate()([v_conv, h_conv])
        if bn:
            x = self.__bn(x)
        x = self.__activation(x, activation=activation)
        return x

    def __detection_layer(self, x, name='sbd_output'):
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
    def __dropout(x, rate):
        return tf.keras.layers.Dropout(rate)(x)

    @staticmethod
    def __drop_filter(x, rate):
        return tf.keras.layers.SpatialDropout2D(rate)(x)

    @staticmethod
    def __bn(x):
        return tf.keras.layers.BatchNormalization(beta_initializer=tf.keras.initializers.zeros(), fused=True)(x)

    @staticmethod
    def __activation(x, activation='none'):
        if activation == 'relu':
            return tf.keras.layers.Activation('relu')(x)
        elif activation == 'silu' or activation == 'swish':
            x_sigmoid = tf.keras.layers.Activation('sigmoid')(x)
            return tf.keras.layers.Multiply()([x, x_sigmoid])
        elif activation == 'mish':
            x_softplus = tf.keras.layers.Activation('softplus')(x)
            x_tanh = tf.keras.layers.Activation('tanh')(x_softplus)
            return tf.keras.layers.Multiply()([x, x_tanh])
        elif activation == 'none':
            return x
        else:
            print(f'[FATAL] unknown activation : [{activation}]')
            exit(-1)

