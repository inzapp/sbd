"""
Authors : inzapp

Github url : https://github.com/inzapp/sbd

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
import tensorflow as tf

from logger import Logger


class Model:
    def __init__(self, cfg, num_classes):
        self.cfg = cfg
        self.num_classes = num_classes
        self.input_shape = (self.cfg.input_rows, self.cfg.input_cols, self.cfg.input_channels)
        self.fused_activations = ['linear', 'relu', 'sigmoid', 'tanh', 'softplus']
        self.available_activations = self.fused_activations + ['leaky', 'silu', 'swish', 'mish']
        self.models = dict()
        self.models['n'] = self.n
        self.models['s'] = self.s
        self.models['m'] = self.m
        self.models['l'] = self.l
        self.models['x'] = self.x
        self.available_backbones = list(self.models.keys())
        self.available_num_output_layers = ['1', 'm']
        self.available_pyramid_scales = [2, 3, 4, 5, 6]
        self.default_model_type = '1p3'

    def is_model_type_valid(self, model_type):
        if type(model_type) != str:
            return False

        model_type = model_type.lower()

        if len(model_type) == 1:
            if model_type in self.available_backbones:
                model_type += self.default_model_type
            else:
                return False

        if len(model_type) != 4:
            return False

        backbone, num_output_layers, p, pyramid_scale = list(model_type)

        if backbone not in self.available_backbones:
            return False

        if num_output_layers not in self.available_num_output_layers:
            return False

        if p != 'p':
            return False

        if not pyramid_scale.isdigit():
            return False

        pyramid_scale = int(pyramid_scale)

        if pyramid_scale not in self.available_pyramid_scales:
            return False

        if num_output_layers == 'm':
            if (self.cfg.p6_model and pyramid_scale == 6) or (not self.cfg.p6_model and pyramid_scale == 5):
                valid_type = f'{backbone}1p{pyramid_scale}'
                Logger.error(f'{model_type} is same with {valid_type}, change model type to {valid_type} for clear usage')

        if pyramid_scale == 6 and not self.cfg.p6_model:
            Logger.warn('6 pyramid scale is only support with p6 model, change p6_model to true in cfg file, model will be built with 5 pyramid scale')

        return backbone, num_output_layers, pyramid_scale

    def build(self, strategy, optimizer, model_type):
        valid = self.is_model_type_valid(model_type)
        if not valid:
            Logger.error([
                f'invalid model type => {model_type}',
                f'model type must be in {self.available_backbones} or custom model type',
                f'',
                r'custom model type is string for combination of "{backbone}{num_output_layers}{p}{pyramid_scale}"',
                f'  backbone : {self.available_backbones}',
                f'  num_output_layers : {self.available_num_output_layers}, m for multi output layers for given pyramid scale',
                f'  p : constant character p for naming rule',
                f'  pyramid_scale : scale value for feature pyramid. max resolution scale of output layers',
                f'',
                f'  if use default model type{self.available_backbones} model will be built with {self.default_model_type} type as default',
                f'',
                f'  ex) n1p3 : nano backbone with one output layer, 3 pyramid scale : output layer resolution is divided by 8(2^3) of input resolution',
                f'  ex) smp4 : small backbone with multi output layer(2 output layer for pyramid sacle 4), 4 pyramid scale : output layer resolution is divided by 16(2^4) of input resolution',
                f'  ex) xmp2 : x-large backbone with multi output layer(4 output layer for pyramid sacle 4), 4 pyramid scale : output layer resolution is divided by 4(2^2) of input resolution'])
        backbone, num_output_layers, pyramid_scale = valid
        with strategy.scope():
            model = self.models[backbone](num_output_layers, pyramid_scale)
            model.compile(optimizer=optimizer)
        return model

    def n(self, num_output_layers, pyramid_scale):
        layer_infos = [
            ['conv', 3, 8, 1],
            ['conv', 3, 16, 1],
            ['conv', 3, 32, 2],
            ['csp',  3, 64, 3],
            ['csp',  3, 128, 3],
            ['csp',  3, 256, 3],
            ['csp',  3, 256, 3],
            ['head', -1, -1, -1],
        ]
        return self.build_layers(layer_infos, num_output_layers, pyramid_scale)

    def s(self, num_output_layers, pyramid_scale):
        layer_infos = [
            ['conv', 3, 16, 1],
            ['conv', 3, 32, 1],
            ['conv', 3, 64, 2],
            ['csp',  3, 128, 3],
            ['csp',  3, 256, 4],
            ['csp',  3, 512, 5],
            ['csp',  3, 512, 5],
            ['head', -1, -1, -1],
        ]
        return self.build_layers(layer_infos, num_output_layers, pyramid_scale)

    def m(self, num_output_layers, pyramid_scale):
        layer_infos = [
            ['conv', 3, 16, 1],
            ['conv', 3, 32, 1],
            ['conv', 3, 96, 2],
            ['csp',  3, 192, 3],
            ['csp',  3, 384, 5],
            ['csp',  3, 512, 7],
            ['csp',  3, 512, 7],
            ['head', -1, -1, -1],
        ]
        return self.build_layers(layer_infos, num_output_layers, pyramid_scale)

    def l(self, num_output_layers, pyramid_scale):
        layer_infos = [
            ['conv', 3, 24, 1],
            ['conv', 3, 48, 2],
            ['conv', 3, 96, 2],
            ['csp',  3, 192, 4],
            ['csp',  3, 384, 6],
            ['csp',  3, 768, 8],
            ['csp',  3, 768, 8],
            ['head', -1, -1, -1],
        ]
        return self.build_layers(layer_infos, num_output_layers, pyramid_scale)

    def x(self, num_output_layers, pyramid_scale):
        layer_infos = [
            ['conv', 3, 32, 1],
            ['conv', 3, 64, 2],
            ['conv', 3, 128, 3],
            ['csp',  3, 256, 4],
            ['csp',  3, 512, 6],
            ['csp',  3, 1024, 8],
            ['csp',  3, 1024, 8],
            ['head', -1, -1, -1],
        ]
        return self.build_layers(layer_infos, num_output_layers, pyramid_scale)

    def build_layers(self, layer_infos, num_output_layers, pyramid_scale):
        assert len(layer_infos) == 8 and layer_infos[-1][0] == 'head'
        features = []
        if not self.cfg.p6_model:
            layer_infos.pop(6)
        input_layer = tf.keras.layers.Input(shape=self.input_shape, name='sbd_input')
        x = input_layer
        for i, (method, kernel_size, channel, depth) in enumerate(layer_infos):
            depth -= 1
            if method in ['conv', 'csp']:
                if i > 2:
                    x = self.dropout(x)
                if method == 'conv':
                    for _ in range(depth):
                        x = self.conv2d(x, channel, kernel_size, self.cfg.activation)
                else:
                    x = self.csp_block(x, channel, kernel_size, depth, self.cfg.activation)
                features.append(x)
            elif method == 'head':
                if self.cfg.p6_model:
                    num_upscaling = 6 - pyramid_scale
                    num_upscaling_spp = 4
                else:
                    num_upscaling = 5 - pyramid_scale
                    num_upscaling_spp = 3
                # x = self.spp_block(x, list(reversed(features))[1:num_upscaling_spp+1], self.cfg.activation)
                if num_upscaling > 0:
                    ms = list(reversed([v[0] for v in layer_infos]))[2:num_upscaling+2]
                    ks = list(reversed([v[1] for v in layer_infos]))[2:num_upscaling+2]
                    cs = list(reversed([v[2] for v in layer_infos]))[2:num_upscaling+2]
                    ds = list(reversed([v[3] for v in layer_infos]))[2:num_upscaling+2]
                    fs = list(reversed(features))[1:num_upscaling+1]
                    x = self.fpn_block(x, ms, fs, cs, ks, ds, self.cfg.activation, return_layers=num_output_layers == 'm')
                if type(x) is not list:
                    x = [x]
            else:
                Logger.error(f'invalid layer info method : {method}, available method : [conv, csp, head]')
            if i < (6 if self.cfg.p6_model else 5):
                x = self.conv2d(x, channel, kernel_size, self.cfg.activation, strides=2)

        output_layers = []
        for i in range(len(x)):
            output_layers.append(self.detection_layer(x[i], f'sbd_output_{i}'))
        return tf.keras.models.Model(input_layer, output_layers if type(output_layers) is list else output_layers[0])

    def spatial_attention_block(self, x, activation, bn=False, reduction_ratio=16):
        input_layer = x
        input_filters = input_layer.shape[-1]
        reduced_channel = input_filters // reduction_ratio
        if reduced_channel < 4:
            reduced_channel = 4
        x = self.conv2d(x, reduced_channel, 1, bn=bn, activation=activation)
        x = self.conv2d(x, reduced_channel, 7, bn=bn, activation=activation)
        x = self.conv2d(x, input_filters, 1, bn=bn, activation='sigmoid')
        return tf.keras.layers.Multiply()([x, input_layer])

    def spp_block(self, x, layers, activation, bn=False):
        pool_size = 2
        spp_layers = [x]
        channels = x.shape[-1]
        for i in range(len(layers)):
            spp_x = self.maxpool2d(layers[i], pool_size=pool_size)
            spp_x = self.conv2d(spp_x, channels, 1, bn=bn, activation=activation)
            spp_layers.append(spp_x)
            pool_size *= 2
        return self.conv2d(self.add(spp_layers), channels, 1, bn=bn, activation=activation)

    def fpn_block(self, x, methods, layers, filters, kernel_sizes, depths, activation, bn=False, return_layers=False, mode='add'):
        assert mode in ['add', 'concat']
        output_layers = [x]
        for i in range(len(layers)):
            if mode == 'add':
                x = self.conv2d(x, filters[i], 1, bn=bn, activation=activation)
                x = self.upsampling2d(x)
                x = self.add([x, layers[i]])
            else:
                x = self.upsampling2d(x)
                x = self.concat([x, layers[i]])
                x = self.conv2d(x, filters[i], 1, bn=bn, activation=activation)
            x = self.bn(x)
            if methods[i] == 'conv':
                for _ in range(depths[i]):
                    x = self.conv2d(x, filters[i], kernel_sizes[i], activation=activation)
            elif methods[i] == 'csp':
                x = self.csp_block(x, filters[i], kernel_sizes[i], depth=depths[i], activation=activation)
            else:
                Logger.error(f'invalid layer info method : {methods[i]}')
            output_layers.append(x)
        return output_layers if return_layers else x

    def csp_block(self, x, filters, kernel_size, depth, activation, bn=False):
        half_filters = filters // 2
        x_0 = self.conv2d(x, half_filters, 1, bn=bn, activation=activation)
        x_1 = self.conv2d(x, filters, 1, bn=bn, activation=activation)
        for _ in range(depth):
            x_0 = self.conv2d(x_0, half_filters, kernel_size, bn=bn, activation=activation)
        x_0 = self.conv2d(x_0, filters, 1, bn=bn, activation=activation)
        x = self.add([x_0, x_1])
        x = self.bn(x)
        x = self.conv2d(x, filters, 1, bn=bn, activation=activation)
        return x

    def conv2d(self, x, filters, kernel_size, activation, strides=1, bn=False):
        assert activation in self.available_activations, f'activation must be one of {self.available_activations}'
        fuse_activation = False if bn else activation in self.fused_activations
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            bias_initializer=self.bias_initializer(),
            activation=activation if fuse_activation else 'linear',
            padding='same',
            use_bias=not bn,
            kernel_regularizer=self.kernel_regularizer())(x)
        if bn:
            x = self.bn(x)
        if not fuse_activation:
            x = self.act(x, activation=activation)
        return x

    def detection_layer(self, x, name='sbd_output'):
        return tf.keras.layers.Conv2D(
            filters=self.num_classes + 5,
            kernel_size=1,
            activation='sigmoid',
            name=name)(x)

    def act(self, x, activation='linear'):
        if activation in self.fused_activations:
            return tf.keras.layers.Activation(activation)(x)
        elif activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        elif activation in ['silu', 'swish']:
            x_sigmoid = tf.keras.layers.Activation('sigmoid')(x)
            return self.multiply([x, x_sigmoid])
        elif activation == 'mish':
            x_softplus = tf.keras.layers.Activation('softplus')(x)
            softplus_tanh = tf.keras.layers.Activation('tanh')(x_softplus)
            return self.multiply([x, softplus_tanh])
        else:
            Logger.error(f'unknown activation : [{activation}]')

    def bn(self, x):
        return tf.keras.layers.BatchNormalization(beta_initializer=self.bias_initializer(), fused=True)(x)

    def kernel_initializer(self):
        return tf.keras.initializers.glorot_normal()

    def bias_initializer(self):
        return tf.keras.initializers.zeros()

    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(l2=self.cfg.l2) if self.cfg.l2 > 0.0 else None

    def dropout(self, x):
        return tf.keras.layers.Dropout(self.cfg.dropout)(x) if self.cfg.dropout > 0.0 else x

    @staticmethod
    def maxpool2d(x, pool_size=2):
        return tf.keras.layers.MaxPool2D(pool_size=(pool_size, pool_size))(x)

    @staticmethod
    def upsampling2d(x):
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

