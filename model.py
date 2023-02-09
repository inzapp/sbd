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
        self.drop_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
        self.models = dict()
        self.models['lightnet_nano'] = self.lightnet_nano
        self.models['lightnet_s'] = self.lightnet_s
        self.models['lightnet_s_csp'] = self.lightnet_s_csp
        self.models['lightnet_m'] = self.lightnet_m
        self.models['lightnet_l'] = self.lightnet_l
        self.models['lightnet_x'] = self.lightnet_x
        self.models['lpd_crop'] = self.lpd_crop
        self.models['lcd'] = self.lcd
        self.models['normal_model'] = self.normal_model
        self.models['lpd_v1'] = self.lpd_v1
        self.models['lpd_v2'] = self.lpd_v2

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def build(self, model_type):
        try:
            return self.models[model_type]()
        except KeyError:
            ModelUtil.print_error_exit([
                f'invalid model type => \'{model_type}\'',
                f'available model types : {list(self.models.keys())}'])

    """
    shape : (384, 640, 1)
    GFLOPs : 3.2095
    parameters : 1,608,294
    forwarding time in cv22  nx8x : 12ms -> need retest
    forwarding time in cv22 16x8x : 21ms -> need retest
    """
    def lightnet_nano(self):
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

        x = self.fpn_block([f0, f1, f2], [64, 128, 256], activation='relu', channel_reduction=False)
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 10.2262
    parameters : 4,725,830
    forwarding time in cv22  nx8x : 20ms -> need retest
    forwarding time in cv22 16x8x : 37ms -> need retest
    """
    def lightnet_s(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x,  64, 1, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f2 = x

        x = self.fpn_block([f0, f1, f2], [128, 256, 512], activation='relu', channel_reduction=True)
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 10.2262
    parameters : 4,725,830
    forwarding time in cv22  nx8x : 20ms -> need retest
    forwarding time in cv22 16x8x : 37ms -> need retest
    """
    def lightnet_s_csp(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.csp_block_new(x, 128, 3, depth=3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.csp_block_new(x, 256, 3, depth=3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.csp_block_new(x, 512, 3, depth=3, activation='relu')
        f2 = x

        x = self.csp_fpn_block([f0, f1, f2], [128, 256, 512], depth=3, activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 19.6136
    parameters : 10,460,860
    forwarding time in cv22  nx8x : 28ms -> need retest
    forwarding time in cv22 16x8x : 51ms -> need retest
    """
    def lightnet_m(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 192, 3, activation='relu')
        x = self.conv_block(x,  96, 1, activation='relu')
        x = self.conv_block(x, 192, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 384, 3, activation='relu')
        x = self.conv_block(x, 192, 1, activation='relu')
        x = self.conv_block(x, 384, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 768, 3, activation='relu')
        x = self.conv_block(x, 384, 1, activation='relu')
        x = self.conv_block(x, 758, 3, activation='relu')
        f2 = x

        x = self.fpn_block([f0, f1, f2], [192, 384, 768], activation='relu', channel_reduction=True)
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 32.6980
    parameters : 18,566,470
    forwarding time in cv22  nx8x : 56ms -> need retest
    forwarding time in cv22 16x8x : 105ms -> need retest
    """
    def lightnet_l(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 1024, 3, activation='relu')
        x = self.conv_block(x,  512, 1, activation='relu')
        x = self.conv_block(x, 1024, 3, activation='relu')
        f2 = x

        x = self.fpn_block([f0, f1, f2], [256, 512, 1024], activation='relu', channel_reduction=True)
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    shape : (384, 640, 1)
    GFLOPs : 45.2446
    parameters : 18,930,886
    forwarding time in cv22  nx8x : 95ms -> need retest
    forwarding time in cv22 16x8x : 178ms -> need retest
    """
    def lightnet_x(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.max_pool(x)

        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv_block(x, 1024, 3, activation='relu')
        x = self.conv_block(x,  512, 1, activation='relu')
        x = self.conv_block(x, 1024, 3, activation='relu')
        f2 = x

        x = self.fpn_block([f0, f1, f2], [256, 512, 1024], activation='relu', channel_reduction=True)
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    size : 640x384x1
    GFLOPs : 37.5203
    parameters : 15,574,481
    forwarding time in cv22  nx8x : 67ms
    forwarding time in cv22 16x8x : 
    """
    def normal_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = input_layer
        x = self.conv_block(x, 24, 3, activation='relu')
        x = self.conv_block(x, 24, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[0])
        x = self.conv_block(x, 48, 3, activation='relu')
        x = self.conv_block(x, 48, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[1])
        x = self.conv_block(x, 96, 3, activation='relu')
        x = self.conv_block(x, 96, 3, activation='relu')
        x = self.conv_block(x, 96, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[2])
        x = self.conv_block(x, 192, 3, activation='relu')
        x = self.conv_block(x, 192, 3, activation='relu')
        x = self.conv_block(x, 192, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[3])
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 128, 1, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[4])
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f2 = x

        x = self.pa_block([f0, f1, f2], [192, 256, 512], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    """
    size : 640x384x1
    GFLOPs : 
    parameters : 
    forwarding time in cv22  nx8x : 
    forwarding time in cv22 16x8x : 
    """
    def normal_model2(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = input_layer
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[0])
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[1])
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[2])
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.conv_block(x, 256, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[3])
        x = self.conv_block(x, 384, 3, activation='relu')
        x = self.conv_block(x, 384, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 384, 3, activation='relu')
        x = self.conv_block(x, 384, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[4])
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 256, 1, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        x = self.conv_block(x, 512, 3, activation='relu')
        f2 = x

        x = self.pa_block([f0, f1, f2], [256, 384, 512], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    # (368, 640, 1) cv22 12ms
    def lpd_v1(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 8, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[0])
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[1])
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[1])
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[2])
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[2])
        x = self.conv_block(x, 64, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[3])
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[3])
        x = self.conv_block(x, 128, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[4])
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[4])
        x = self.conv_block(x, 256, 3, activation='relu')
        f2 = x

        x = self.fpn_block([f0, f1, f2], [64, 128, 256], activation='relu')
        y = self.detection_layer(x, 'sbd_output')
        return tf.keras.models.Model(input_layer, y)

    # (352, 640, 1) cv22 20ms (16x 8x)
    def lpd_v2(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[0])
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[1])
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[1])
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.csp_block(x, 128, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu', drop_rate=self.drop_rates[2])
        x = self.max_pool(x)
        f1 = x

        x = self.csp_block(x, 256, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu', drop_rate=self.drop_rates[3])
        x = self.max_pool(x)

        x = self.csp_block(x, 512, 3, first_depth_n_convs=1, second_depth_n_convs=4, activation='relu', inner_activation='relu', drop_rate=self.drop_rates[4])
        f2 = x

        x = self.fpn_block([f1, f2], [256, 256], activation='relu')
        y = self.detection_layer(x)
        return tf.keras.models.Model(input_layer, y)

    # (320, 320, 1) cv22 12ms
    def lpd_crop(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv_block(input_layer, 8, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[0])
        x = self.conv_block(x, 16, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[1])
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[1])
        x = self.conv_block(x, 32, 3, activation='relu')
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[2])
        x = self.conv_block(x, 64, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[2])
        x = self.conv_block(x, 64, 3, activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[3])
        x = self.conv_block(x, 128, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[3])
        x = self.conv_block(x, 128, 3, activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.dropout(x, self.drop_rates[4])
        x = self.conv_block(x, 256, 3, activation='relu')
        x = self.dropout(x, self.drop_rates[4])
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

    def csp_fpn_block(self, layers, filters, depth, activation, bn=False, return_layers=False):
        assert type(layers) == list and type(filters) == list
        layers = list(reversed(layers))
        ret = [layers[0]]
        filters = list(reversed(filters))
        for i in range(len(layers) - 1):
            x = tf.keras.layers.UpSampling2D()(layers[i] if i == 0 else x)
            x = self.concat([x, layers[i+1]])
            x = self.conv_block(x, filters[i+1], 1, bn=bn, activation=activation)
            x = self.csp_block_new(x, filters[i+1], 3, depth=depth, activation=activation)
            ret.append(x)
        layers = list(reversed(ret))
        return layers if return_layers else x

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

    def csp_block(self, x, filters, kernel_size, drop_rate, first_depth_n_convs=1, second_depth_n_convs=2, bn=False, activation='none', inner_activation='none'):
        half_filters = filters / 2
        x_0 = self.conv_block(x, half_filters, 1, activation='none')
        for i in range(first_depth_n_convs):
            if i == 0:
                x_1 = self.conv_block(x, half_filters, 1, activation=inner_activation)
            else:
                x_1 = self.dropout(x_1, drop_rate * 0.5)
                x_1 = self.conv_block(x_1, half_filters, kernel_size, activation=inner_activation)
        x_1_0 = self.conv_block(x_1, half_filters, 1, activation='none')
        for i in range(second_depth_n_convs):
            if i == 0:
                x_1_1 = self.conv_block(x_1, half_filters, 1, activation=inner_activation)
            else:
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
    def dropout(x, rate):
        return x
        # return tf.keras.layers.Dropout(rate)(x)

