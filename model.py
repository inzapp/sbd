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

from util import ModelUtil


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


@tf.keras.utils.register_keras_serializable(package='Custom', name='WeightStandardization')
class WeightStandardization(tf.keras.regularizers.Regularizer):
  def __init__(self):
      super().__init__()

  def __call__(self, x):
      x -= tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True)
      x /= tf.math.reduce_std(x, axis=[0, 1, 2], keepdims=True) + 1e-5
      return x

  def get_config(self):
      return {}


class Model:
    def __init__(self, input_shape, output_channel, l2):
        self.input_shape = input_shape
        self.output_channel = output_channel
        self.l2 = l2
        self.models = dict()
        self.models['n1'] = self.n
        self.models['n3'] = self.n
        self.models['n5'] = self.n
        self.models['s1'] = self.s
        self.models['s3'] = self.s
        self.models['s5'] = self.s
        self.models['m1'] = self.m
        self.models['m3'] = self.m
        self.models['m5'] = self.m
        self.models['l1'] = self.l
        self.models['l3'] = self.l
        self.models['l5'] = self.l
        self.models['x1'] = self.x
        self.models['x3'] = self.x
        self.models['x5'] = self.x
        self.models['lpd_crop'] = self.lpd_crop
        self.models['lcd'] = self.lcd
        self.models['lpd_v1'] = self.lpd_v1
        self.models['lpd_v2'] = self.lpd_v2

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def build(self, model_type):
        try:
            return self.models[model_type](int(model_type[1]))
        except KeyError:
            ModelUtil.print_error_exit([
                f'invalid model type => \'{model_type}\'',
                f'available model types : {list(self.models.keys())}'])

    """
    shape : (384, 640, 1)
    GFLOPs : 1.9766
    parameters : 947,815
    forwarding time in CV22
        nx8x : 4.78ms
        16x8x : 8.11ms
        16x16x : 15.08ms
    """
    def n(self, num_output_layers):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 8, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 64, 3, depth=3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 128, 3, depth=3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 256, 3, depth=3, activation='relu')

        f2, f1, f0 = self.csp_fpn_block(x, [f1, f0], [128, 64], depth=3, activation='relu', return_layers=True)
        y = self.detection_block(f0, f1, f2, num_output_layers)
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 7.8067
    parameters : 3,785,159
    forwarding time in CV22
        nx8x : 15.28ms
        16x8x : 28.12ms
        16x16x : 55.61ms
    """
    def s(self, num_output_layers):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 128, 3, depth=3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 256, 3, depth=3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 512, 3, depth=3, activation='relu')

        f2, f1, f0 = self.csp_fpn_block(x, [f1, f0], [256, 128], depth=3, activation='relu', return_layers=True)
        y = self.detection_block(f0, f1, f2, num_output_layers)
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 16.0880
    parameters : 7,113,927
    forwarding time in CV22
        nx8x : 29.91ms
        16x8x : 55.85ms
        16x16x : 111.71ms
    """
    def m(self, num_output_layers):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 192, 3, depth=4, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 384, 3, depth=4, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 512, 3, depth=4, activation='relu')

        f2, f1, f0 = self.csp_fpn_block(x, [f1, f0], [384, 192], depth=4, activation='relu', return_layers=True)
        y = self.detection_block(f0, f1, f2, num_output_layers)
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 24.6675
    parameters : 9,319,495
    forwarding time in CV22
        nx8x : 44.38ms
        16x8x : 84.73ms
        16x16x : 169.54ms
    """
    def l(self, num_output_layers):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 256, 3, depth=5, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 384, 3, depth=5, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 512, 3, depth=5, activation='relu')

        f2, f1, f0 = self.csp_fpn_block(x, [f1, f0], [384, 256], depth=5, activation='relu', return_layers=True)
        y = self.detection_block(f0, f1, f2, num_output_layers)
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 46.8271
    parameters : 14,740,679
    forwarding time in CV22
        nx8x : 82.97ms
        16x8x : 160.21ms
        16x16x : 318.20ms
    """
    def x(self, num_output_layers):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 256, 3, depth=6, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 512, 3, depth=6, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.csp_block_light(x, 512, 3, depth=6, activation='relu')

        f2, f1, f0 = self.csp_fpn_block(x, [f1, f0], [512, 256], depth=6, activation='relu', return_layers=True)
        y = self.detection_block(f0, f1, f2, num_output_layers)
        return tf.keras.models.Model(input_layer, y)

    # (368, 640, 1) CV22 12ms
    def lpd_v1(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 8, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f2 = x

        x = self.fpn_block([f0, f1, f2], [64, 128, 256], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    # (352, 640, 1) CV22 20ms (16x 8x)
    def lpd_v2(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.csp_block(x, 128, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu')
        x = self.max_pool(x)
        f1 = x

        x = self.csp_block(x, 256, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu')
        x = self.max_pool(x)

        x = self.csp_block(x, 512, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu')
        f2 = x

        x = self.fpn_block([f1, f2], [256, 256], activation='relu')
        y = self.detection_layer(x)
        return tf.keras.models.Model(input_layer, y)

    # (320, 320, 1) CV22 12ms
    def lpd_crop(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 8, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f2 = x

        x = self.fpn_block([f0, f1, f2], [64, 128, 256], activation='relu')
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

        x = self.fpn_block([f0, f1, f2], [128, 256, 512], activation='relu', additional_conv=False)
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    def spatial_attention_block(self, x, activation, bn=False, reduction_ratio=16):
        input_layer = x
        input_filters = input_layer.shape[-1]
        reduced_channel = input_filters // reduction_ratio
        if reduced_channel < 4:
            reduced_channel = 4
        x = self.conv_block(x, reduced_channel, 1, bn=bn, activation=activation)
        x = self.conv_block(x, reduced_channel, 7, bn=bn, activation=activation)
        x = self.conv_block(x, input_filters, 1, bn=bn, activation='sigmoid')
        return tf.keras.layers.Multiply()([x, input_layer])

    def fpn_block(self, layers, filters, activation, bn=False, return_layers=False, channel_reduction=True, additional_conv=True):
        assert type(layers) == list and type(filters) == list
        layers = list(reversed(layers))
        ret = [layers[0]]
        filters = list(reversed(filters))
        for i in range(len(layers) - 1):
            x = tf.keras.layers.UpSampling2D()(layers[i] if i == 0 else x)
            x = self.concat([x, layers[i+1]])
            x = self.conv_block(x, filters[i+1], 1, bn=bn, activation=activation)
            x = self.conv_block(x, filters[i+1], 3, bn=bn, activation=activation)
            if additional_conv:
                if channel_reduction:
                    x = self.conv_block(x, filters[i+1] // 2, 1, bn=bn, activation=activation)
                x = self.conv_block(x, filters[i+1], 3, bn=bn, activation=activation)
            ret.append(x)
        layers = list(reversed(ret))
        return layers if return_layers else x

    def csp_fpn_block(self, x, layers, filters, depth, activation, bn=False, return_layers=False, mode='add'):
        assert mode in ['add', 'concat']
        assert type(layers) == list and type(filters) == list
        output_layers = [x]
        for i in range(len(layers)):
            if mode == 'add':
                x = self.conv_block(x, filters[i], 1, bn=bn, activation=activation)
                x = self.upsampling(x)
                x = self.add([x, layers[i]])
            else:
                x = self.upsampling(x)
                x = self.concat([x, layers[i]])
                x = self.conv_block(x, filters[i], 1, bn=bn, activation=activation)
            x = self.csp_block_light(x, filters[i], 3, depth=depth, activation=activation)
            output_layers.append(x)
        return output_layers if return_layers else x

    # def fpn_block(self, layers, filters, activation, bn=False, return_layers=False):
    #     assert type(layers) == list and type(filters) == list
    #     layers = list(reversed(layers))
    #     ret = [layers[0]]
    #     filters = list(reversed(filters))
    #     for i in range(len(layers) - 1):
    #         x = tf.keras.layers.UpSampling2D()(layers[i] if i == 0 else x)
    #         if filters[i] != filters[i+1]:
    #             x = self.conv_block(x, filters[i+1], 1, bn=bn, activation=activation)
    #         x = self.add([x, layers[i+1]])
    #         x = self.conv_block(x, filters[i+1], 3, bn=bn, activation=activation)
    #         ret.append(x)
    #     layers = list(reversed(ret))
    #     return layers if return_layers else x

    def pa_block(self, layers, filters, activation, bn=False, return_layers=False):
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

    def csp_block(self, x, filters, kernel_size, drop_rate=0.0, first_depth_n_convs=1, second_depth_n_convs=2, bn=False, activation='none', inner_activation='none'):
        half_filters = filters / 2
        x_0 = self.conv_block(x, half_filters, 1, activation='none')
        for i in range(first_depth_n_convs):
            if i == 0:
                x_1 = self.conv_block(x, half_filters, 1, activation=inner_activation)
            else:
                if drop_rate > 0.0:
                    x_1 = self.dropout(x_1, drop_rate * 0.5)
                x_1 = self.conv_block(x_1, half_filters, kernel_size, activation=inner_activation)
        x_1_0 = self.conv_block(x_1, half_filters, 1, activation='none')
        for i in range(second_depth_n_convs):
            if i == 0:
                x_1_1 = self.conv_block(x_1, half_filters, 1, activation=inner_activation)
            else:
                if drop_rate > 0.0:
                    x_1_1 = self.dropout(x_1_1, drop_rate * 0.5)
                x_1_1 = self.conv_block(x_1_1, half_filters, kernel_size, activation='none' if i == second_depth_n_convs - 1 else inner_activation)
        x_1 = tf.keras.layers.Concatenate()([x_1_0, x_1_1])
        x = tf.keras.layers.Concatenate()([x_0, x_1])
        if bn:
            x = self.bn(x)
        x = self.activation(x, activation=activation)
        x = self.conv_block(x, filters, 1, bn=bn, activation=activation)
        return x

    def csp_block_new(self, x, filters, kernel_size, depth, bn=False, activation='none'):
        half_filters = filters / 2
        x_0 = self.conv_block(x, half_filters, 1, bn=bn, activation=activation)
        x_1 = self.conv_block(x, half_filters, 1, bn=bn, activation=activation)
        for _ in range(depth):
            x_0_1 = self.conv_block(x_0, half_filters, 1, bn=bn, activation=activation)
            x_0_1 = self.conv_block(x_0_1, half_filters, kernel_size, bn=bn, activation=activation)
            x_0 = self.add([x_0, x_0_1])
        x = self.concat([x_0, x_1])
        # x = self.conv_block(x, filters, 1, bn=bn, activation=activation)
        return x

    def csp_block_light(self, x, filters, kernel_size, depth, bn=False, activation='none'):
        half_filters = filters / 2
        x_0 = self.conv_block(x, half_filters, 1, bn=bn, activation=activation)
        x_1 = self.conv_block(x, half_filters, 1, bn=bn, activation=activation)
        for _ in range(depth):
            x_0 = self.conv_block(x_0, half_filters, kernel_size, bn=bn, activation=activation)
        x = self.concat([x_0, x_1])
        x = self.conv_block(x, filters, 1, bn=bn, activation=activation)
        return x

    def csp_block_light_2_depth(self, x, filters, kernel_size, depth, bn=False, activation='none'):
        half_filters = filters // 2
        quarter_filters = filters // 4
        x_0 = self.conv_block(x, half_filters, 1, bn=bn, activation=activation)
        x_1 = self.conv_block(x, half_filters, 1, bn=bn, activation=activation)
        for _ in range(depth):
            x_1 = self.conv_block(x_1, half_filters, kernel_size, bn=bn, activation=activation)
        x_1_1 = self.conv_block(x_1, quarter_filters, 1, bn=bn, activation=activation)
        for _ in range(depth):
            x_1_1 = self.conv_block(x_1_1, quarter_filters, kernel_size, bn=bn, activation=activation)
        x_1 = self.concat([x_1, x_1_1])
        x_1 = self.conv_block(x_1, filters, 1, bn=bn, activation=activation)
        x = self.concat([x_0, x_1])
        x = self.conv_block(x, filters, 1, bn=bn, activation=activation)
        return x

    def conv_block(self, x, filters, kernel_size, bn=False, activation='none'):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            bias_initializer=self.bias_initializer(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=self.kernel_regularizer())(x)
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
            kernel_initializer=self.kernel_initializer(),
            bias_initializer=self.bias_initializer(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=self.kernel_regularizer())(x)
        # if mode == 'stack':
        #     if bn:
        #         v_conv = self.bn(v_conv)
        #     v_conv = self.activation(v_conv, activation=activation)
        h_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, kernel_size),
            kernel_initializer=self.kernel_initializer(),
            bias_initializer=self.bias_initializer(),
            padding='same',
            use_bias=False if bn else True,
            kernel_regularizer=self.kernel_regularizer())(v_conv if mode == 'stack' else x)
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

    def detection_block(self, f0, f1, f2, num_output_layers):
        x = []
        if num_output_layers == 1:
            x = self.detection_layer(f0, 'sbd_output_00')
        elif num_output_layers == 3:
            x.append(self.detection_layer(f0, 'sbd_output_00'))
            x.append(self.detection_layer(f1, 'sbd_output_10'))
            x.append(self.detection_layer(f2, 'sbd_output_20'))
        elif num_output_layers == 5:
            x.append(self.detection_layer(f0, 'sbd_output_00'))
            x.append(self.detection_layer(f1, 'sbd_output_10'))
            x.append(self.detection_layer(f1, 'sbd_output_11'))
            x.append(self.detection_layer(f2, 'sbd_output_20'))
            x.append(self.detection_layer(f2, 'sbd_output_21'))
        else:
            ModelUtil.print_error_exit(f'invalid num_output_layers : {num_output_layers}')
        return x

    def detection_layer(self, x, name='sbd_output'):
        return tf.keras.layers.Conv2D(
            filters=self.output_channel,
            kernel_size=1,
            activation='sigmoid',
            name=name)(x)

    def activation(self, x, activation='none'):
        if activation in ['relu', 'sigmoid', 'tanh']:
            return tf.keras.layers.Activation(activation)(x)
        elif activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        elif activation == 'silu' or activation == 'swish':
            x_sigmoid = tf.keras.layers.Activation('sigmoid')(x)
            return self.multiply([x, x_sigmoid])
        elif activation == 'mish':
            x_softplus = tf.keras.layers.Activation('softplus')(x)
            softplus_tanh = tf.keras.layers.Activation('tanh')(x_softplus)
            return self.multiply([x, softplus_tanh])
        elif activation == 'none':
            return x
        else:
            print(f'[FATAL] unknown activation : [{activation}]')
            exit(-1)

    def bn(self, x):
        return tf.keras.layers.BatchNormalization(beta_initializer=self.bias_initializer(), fused=True)(x)

    def kernel_initializer(self):
        return tf.keras.initializers.glorot_normal()

    def bias_initializer(self):
        return tf.keras.initializers.zeros()

    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(l2=self.l2) if self.l2 > 0.0 else None

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
    def dropout(x):
        return tf.keras.layers.Dropout(0.1)(x)

