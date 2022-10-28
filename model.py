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
    def __init__(self, input_shape, output_channel, decay, drop_rate=0.0625):
        self.input_shape = input_shape
        self.output_channel = output_channel
        self.drop_rate = drop_rate
        self.decay = decay

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def build(self):
        # return self.student()
        # return self.teacher()
        return self.lightnet_nano()
        # return self.lightnet_s()
        # return self.lightnet_m()
        # return self.lightnet_l()
        # return self.lightnet_x()
        # return self.lightnet_illusion()
        # return self.lpd_v1()
        # return self.lpd_v2()
        # return self.lpd_crop()
        # return self.lcd()
        # return self.vgg_16()
        # return self.darknet_19()

    """
    size : 640x384x1
    GFLOPs : 6.0208
    parameters : 5,558,438
    forwarding time in cv2  nx8x : 12ms
    forwarding time in cv2 16x8x : 21ms
    """
    def lightnet_nano(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.cross_conv_block(input_layer, 16, 3, mode='concat', activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.cross_conv_block(x, 32, 3, mode='concat', activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.cross_conv_block(x, 64, 3, mode='concat', activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.cross_conv_block(x, 128, 3, mode='concat', activation='relu')
        x = self.cross_conv_block(x, 128, 3, mode='concat', activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f1, f2], [256, 512], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    size : 640x384x1
    GFLOPs : 10.5687
    parameters : 6,298,558
    forwarding time in cv2  nx8x : 20ms
    forwarding time in cv2 16x8x : 37ms
    """
    def lightnet_s(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = input_layer
        x = self.cross_conv_block(x, 16, 3, mode='concat', activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.cross_conv_block(x, 32, 3, mode='concat', activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.spatial_attention_block(x, activation='relu')

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x,  64, 1, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x,  64, 1, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f0, f1, f2], [128, 256, 512], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    TODO : retest
    size : 640x384x1
    GFLOPs : 14.7717
    parameters : 9,745,054
    forwarding time in cv2  nx8x : 28ms
    forwarding time in cv2 16x8x : 51ms
    """
    def lightnet_m(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = input_layer
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.spatial_attention_block(x, activation='relu')

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x,  64, 1, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x,  64, 1, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f2 = x

        x = self.path_aggregation_network([f0, f1, f2], [128, 256, 512], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    TODO : re test

    size : 640x384x1
    GFLOPs : 29.8480
    parameters : 21,787,550
    forwarding time in cv2  nx8x : 56ms
    forwarding time in cv2 16x8x : 105ms
    """
    def lightnet_l(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = input_layer
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.conv_block(x, 32, 3, activation='relu')  # added
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.spatial_attention_block(x, activation='relu')

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 192, 3, activation='relu')
        x = self.conv_block(x,  96, 1, activation='relu')
        x = self.conv_block(x, 192, 3, activation='relu')
        x = self.conv_block(x,  96, 1, activation='relu')
        x = self.conv_block(x, 192, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 384, 3, activation='relu')
        x = self.conv_block(x, 192, 1, activation='relu')
        x = self.conv_block(x, 384, 3, activation='relu')
        x = self.conv_block(x, 192, 1, activation='relu')
        x = self.conv_block(x, 384, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 768, 3, activation='relu')
        x = self.conv_block(x, 384, 1, activation='relu')
        x = self.conv_block(x, 768, 3, activation='relu')
        x = self.conv_block(x, 384, 1, activation='relu')
        x = self.conv_block(x, 768, 3, activation='relu')
        f2 = x

        x = self.path_aggregation_network([f0, f1, f2], [192, 384, 768], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    size : 640x384x1
    GFLOPs : 50.8383
    parameters : 38,630,558
    forwarding time in cv2  nx8x : 95ms
    forwarding time in cv2 16x8x : 178ms
    """
    def lightnet_x(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.conv_block(x, 32, 3, activation='relu')  # added
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.spatial_attention_block(x, activation='relu')

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 1024, 3, activation='relu')
        x = self.conv_block(x,  512, 1, activation='relu')
        x = self.conv_block(x, 1024, 3, activation='relu')
        x = self.conv_block(x,  512, 1, activation='relu')
        x = self.conv_block(x, 1024, 3, activation='relu')
        f2 = x

        x = self.path_aggregation_network([f0, f1, f2], [256, 512, 1024], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    def student(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 4, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 8, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 128, 3, activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f0, f1, f2], [32, 64, 128], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    def teacher(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = input_layer
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.max_pool(x)

        x = self.spatial_attention_block(x, activation='relu')

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 1024, 3, activation='relu')
        x = self.conv_block(x,  512, 1, activation='relu')
        x = self.conv_block(x, 1024, 3, activation='relu')
        x = self.conv_block(x,  512, 1, activation='relu')
        x = self.conv_block(x, 1024, 3, activation='relu')
        f2 = x

        x = self.path_aggregation_network([f0, f1, f2], [256, 512, 1024], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    def lightnet_illusion(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = input_layer
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.illusion_block(x, 128, depth=3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.illusion_block(x, 256, depth=4, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.illusion_block(x, 512, depth=5, activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f0, f1, f2], [128, 256, 512], activation='relu')
        y = self.detection_layer(x)
        return tf.keras.models.Model(input_layer, y)

    # (368, 640, 1) cv2 12ms
    def lpd_v1(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 8, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 128, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f0, f1, f2], [64, 128, 256], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    # (352, 640, 1) cv2 20ms (16x 8x)
    def lpd_v2(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.csp_block(x, 128, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu')
        x = self.max_pool(x)
        f1 = x

        x = self.drop_filter(x, self.drop_rate)
        x = self.csp_block(x, 256, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.csp_block(x, 512, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f1, f2], [256, 256], activation='relu')
        y = self.detection_layer(x)
        return tf.keras.models.Model(input_layer, y)

    # (320, 320, 1) cv2 12ms
    def lpd_crop(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 8, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 64, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 128, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.drop_filter(x, self.drop_rate)
        x = self.conv_block(x, 256, 3, activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f0, f1, f2], [64, 128, 256], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    def lcd(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 128, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 256, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 512, 3, activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f0, f1, f2], [128, 256, 512], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    def vgg_16(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 64, 3, bn=True, activation='relu')
        x = self.conv_block(x, 64, 3, bn=True, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        y1 = self.detection_layer(x, name='sbd_output_1')
        x = self.max_pool(x)

        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        y2 = self.detection_layer(x, name='sbd_output_2')
        x = self.max_pool(x)

        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        y3 = self.detection_layer(x, name='sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def darknet_19(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 32, 3, bn=True, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, bn=True, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.conv_block(x, 64, 1, bn=True, activation='relu')
        x = self.conv_block(x, 128, 3, bn=True, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 256, 3, bn=True, activation='relu')
        x = self.conv_block(x, 128, 1, bn=True, activation='relu')
        x = self.conv_block(x, 256, 3, bn=True, activation='relu')
        y1 = self.detection_layer(x, 'sbd_output_1')
        x = self.max_pool(x)

        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.conv_block(x, 256, 1, bn=True, activation='relu')
        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        x = self.conv_block(x, 256, 1, bn=True, activation='relu')
        x = self.conv_block(x, 512, 3, bn=True, activation='relu')
        y2 = self.detection_layer(x, 'sbd_output_2')
        x = self.max_pool(x)

        x = self.conv_block(x, 1024, 3, bn=True, activation='relu')
        x = self.conv_block(x, 512, 1, bn=True, activation='relu')
        x = self.conv_block(x, 1024, 3, bn=True, activation='relu')
        x = self.conv_block(x, 512, 1, bn=True, activation='relu')
        x = self.conv_block(x, 1024, 3, bn=True, activation='relu')
        y3 = self.detection_layer(x, 'sbd_output_3')
        return tf.keras.models.Model(input_layer, [y1, y2, y3])

    def spatial_attention_block(self, x, activation, bn=False, reduction_ratio=16):
        input_layer = x
        if tf.keras.backend.image_data_format() == 'channels_first':
            input_filters = input_layer.shape[1]
        else:
            input_filters = input_layer.shape[-1]
        reduced_channel = input_filters // reduction_ratio
        if reduced_channel < 4:
            reduced_channel = 4
        x = self.conv_block(x, reduced_channel, 1, bn=bn, activation=activation)
        x = self.conv_block(x, reduced_channel, 7, bn=bn, activation=activation)
        x = self.conv_block(x, input_filters, 1, bn=bn, activation='sigmoid')
        return tf.keras.layers.Multiply()([x, input_layer])

    def feature_pyramid_network(self, layers, filters, activation, bn=False, return_layers=False):
        assert type(layers) == list and type(filters) == list
        layers = list(reversed(layers))
        ret = [layers[0]]
        filters = list(reversed(filters))
        for i in range(len(layers) - 1):
            x = tf.keras.layers.UpSampling2D()(layers[i] if i == 0 else x)
            if filters[i] != filters[i+1]:
                x = self.conv_block(x, filters[i+1], 1, bn=bn, activation=activation)
            x = self.add([x, layers[i+1]])
            x = self.conv_block(x, filters[i+1], 3, bn=bn, activation=activation)
            ret.append(x)
        layers = list(reversed(ret))
        return layers if return_layers else x

    def path_aggregation_network(self, layers, filters, activation, bn=False, return_layers=False):
        assert type(layers) == list and type(filters) == list
        layers = list(reversed(layers))

        # upsampling with feature addition
        ret = [layers[0]]
        filters = list(reversed(filters))
        for i in range(len(layers) - 1):
            x = tf.keras.layers.UpSampling2D()(layers[i] if i == 0 else x)
            if filters[i] != filters[i+1]:
                x = self.conv_block(x, filters[i+1], 1, bn=bn, activation=activation)
            x = self.add([x, layers[i+1]])
            x = self.conv_block(x, filters[i+1], 3, bn=bn, activation=activation)
            ret.append(x)
        layers = list(reversed(ret))

        # maxpool with feature addition
        ret = [layers[0]]
        filters = list(reversed(filters))
        for i in range(len(layers) - 1):
            x = tf.keras.layers.MaxPool2D()(layers[i] if i == 0 else x)
            if filters[i] != filters[i+1]:
                x = self.conv_block(x, filters[i+1], 1, bn=bn, activation=activation)
            x = self.add([x, layers[i+1]])
            x = self.conv_block(x, filters[i+1], 3, bn=bn, activation=activation)
            ret.append(x)
        layers = list(reversed(ret))

        # upsampling with feature addition
        ret = [layers[0]]
        filters = list(reversed(filters))
        for i in range(len(layers) - 1):
            x = tf.keras.layers.UpSampling2D()(layers[i] if i == 0 else x)
            if filters[i] != filters[i+1]:
                x = self.conv_block(x, filters[i+1], 1, bn=bn, activation=activation)
            x = self.add([x, layers[i+1]])
            x = self.conv_block(x, filters[i+1], 3, bn=bn, activation=activation)
            ret.append(x)
        layers = list(reversed(ret))
        return layers if return_layers else x

    def illusion_block(self, x, filters, depth=1, bn=False, activation='none'):
        if depth == 0:
            return x
        h_filters = max(filters // 2, 16)
        x_0 = self.conv_block(x, h_filters, 1, bn=bn, activation=activation)
        x_1 = self.conv_block(x, h_filters, 3, bn=bn, activation=activation)
        x_2 = self.conv_block(x, h_filters, 5, bn=bn, activation=activation)
        x_1 = self.illusion_block(x_1, h_filters, depth=depth-1, bn=bn, activation=activation)
        x = self.concat([x_0, x_1, x_2])
        x = self.conv_block(x, filters, 1, bn=bn, activation=activation)
        return x

    def csp_block(self, x, filters, kernel_size, first_depth_n_convs=1, second_depth_n_convs=2, bn=False, activation='none', inner_activation='none'):
        half_filters = filters / 2
        x_0 = self.conv_block(x, half_filters, 1, activation='none')
        for i in range(first_depth_n_convs):
            if i == 0:
                x_1 = self.conv_block(x, half_filters, 1, activation=inner_activation)
            else:
                x_1 = self.drop_filter(x_1, self.drop_rate)
                x_1 = self.conv_block(x_1, half_filters, kernel_size, activation=inner_activation)
        x_1_0 = self.conv_block(x_1, half_filters, 1, activation='none')
        for i in range(second_depth_n_convs):
            if i == 0:
                x_1_1 = self.conv_block(x_1, half_filters, 1, activation=inner_activation)
            else:
                x_1_1 = self.drop_filter(x_1_1, self.drop_rate)
                x_1_1 = self.conv_block(x_1_1, half_filters, kernel_size, activation='none' if i == second_depth_n_convs - 1 else inner_activation)
        x_1 = tf.keras.layers.Concatenate()([x_1_0, x_1_1])
        x = tf.keras.layers.Concatenate()([x_0, x_1])
        if bn:
            x = self.bn(x)
        x = self.activation(x, activation=activation)
        x = self.conv_block(x, filters, 1, bn=bn, activation=activation)
        return x

    def conv_block(self, x, filters, kernel_size, bn=False, activation='none'):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.keras.initializers.zeros(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.decay) if self.decay > 0.0 else None)(x)
        if bn:
            x = self.bn(x)
        x = self.activation(x, activation=activation)
        return x

    def cross_csp_block(self, x, filters, kernel_size, first_depth_n_convs=1, second_depth_n_convs=2, mode='add', bn=False, activation='none', inner_activation='none'):
        half_filters = filters // 2
        x_0 = self.cross_conv_block(x, half_filters, 1, mode=mode, activation='none')
        for i in range(first_depth_n_convs):
            if i == 0:
                x_1 = self.cross_conv_block(x, half_filters, 1, mode=mode, activation=inner_activation)
            else:
                x_1 = self.cross_conv_block(x_1, half_filters, kernel_size, mode=mode, activation=inner_activation)
        x_1_0 = self.cross_conv_block(x_1, half_filters, 1, activation='none')
        for i in range(second_depth_n_convs):
            if i == 0:
                x_1_1 = self.cross_conv_block(x_1, half_filters, 1, mode=mode, activation=inner_activation)
            else:
                x_1_1 = self.cross_conv_block(x_1_1, half_filters, kernel_size, mode=mode, activation='none' if i == second_depth_n_convs - 1 else inner_activation)
        x_1 = tf.keras.layers.Concatenate()([x_1_0, x_1_1])
        x = tf.keras.layers.Concatenate()([x_0, x_1])
        if bn:
            x = self.bn(x)
        x = self.activation(x, activation=activation)
        return x

    def cross_conv_block(self, x, filters, kernel_size, mode='concat', bn=False, activation='none'):
        filters = filters // 2 if mode == 'concat' else filters
        v_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, 1),
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.keras.initializers.zeros(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.decay) if self.decay > 0.0 else None)(x)
        # if mode == 'stack':
        #     if bn:
        #         v_conv = self.bn(v_conv)
        #     v_conv = self.activation(v_conv, activation=activation)
        h_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, kernel_size),
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.keras.initializers.zeros(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.decay) if self.decay > 0.0 else None)(v_conv if mode == 'stack' else x)
        if mode == 'add':
            x = tf.keras.layers.Add()([v_conv, h_conv])
        elif mode == 'concat':
            x = tf.keras.layers.Concatenate()([v_conv, h_conv])
        elif mode == 'stack':
            x = h_conv
        if bn:
            x = self.bn(x)
        x = self.activation(x, activation=activation)
        return x

    def detection_layer(self, x, name='sbd_output'):
        return tf.keras.layers.Conv2D(
            filters=self.output_channel,
            kernel_size=1,
            activation='sigmoid',
            name=name)(x)

    def activation(self, x, activation='none'):
        if activation == 'sigmoid':
            return tf.keras.layers.Activation('sigmoid')(x)
        elif activation == 'tanh':
            return tf.keras.layers.Activation('tanh')(x)
        elif activation == 'relu':
            return tf.keras.layers.Activation('relu')(x)
        elif activation == 'silu' or activation == 'swish':
            x_sigmoid = tf.keras.layers.Activation('sigmoid')(x)
            return self.multiply([x, x_sigmoid])
        elif activation == 'mish':
            x_softplus = tf.keras.layers.Activation('softplus')(x)
            x_tanh = tf.keras.layers.Activation('tanh')(x_softplus)
            return self.multiply([x, x_tanh])
        elif activation == 'none':
            return x
        else:
            print(f'[FATAL] unknown activation : [{activation}]')
            exit(-1)

    @staticmethod
    def max_pool(x):
        return tf.keras.layers.MaxPool2D()(x)

    @staticmethod
    def upsampling(x):
        return tf.keras.layers.UpSampling2D()(x)

    @staticmethod
    def add(layers):
        return tf.keras.layers.Add()(layers)

    @staticmethod
    def multiply(layers):
        return tf.keras.layers.Multiply()(layers)

    @staticmethod
    def concat(layers):
        return tf.keras.layers.Concatenate()(layers)

    @staticmethod
    def bn(x):
        return tf.keras.layers.BatchNormalization(beta_initializer=tf.keras.initializers.zeros(), fused=True)(x)

    @staticmethod
    def drop_filter(x, rate):
        return tf.keras.layers.SpatialDropout2D(rate)(x)

