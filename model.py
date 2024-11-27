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
        stage_infos = [
            ['conv', 8, 3, 1],
            ['conv', 16, 3, 1],
            ['conv', 32, 3, 2],
            ['lcsp', 64, 3, 3],
            ['lcsp', 128, 3, 3],
            ['lcsp', 256, 3, 3],
            ['csp', 256, 3, 1],
        ]
        return self.build_model(stage_infos, num_output_layers, pyramid_scale)

    def s(self, num_output_layers, pyramid_scale):
        stage_infos = [
            ['conv', 16, 3, 1],
            ['conv', 32, 3, 1],
            ['conv', 64, 3, 2],
            ['lcsp', 128, 3, 3],
            ['lcsp', 256, 3, 4],
            ['lcsp', 512, 3, 5],
            ['csp', 512, 3, 2],
        ]
        return self.build_model(stage_infos, num_output_layers, pyramid_scale)

    def m(self, num_output_layers, pyramid_scale):
        stage_infos = [
            ['conv', 16, 3, 1],
            ['conv', 32, 3, 1],
            ['conv', 96, 3, 2],
            ['lcsp', 192, 3, 3],
            ['lcsp', 384, 3, 5],
            ['lcsp', 512, 3, 7],
            ['csp', 512, 3, 3],
        ]
        return self.build_model(stage_infos, num_output_layers, pyramid_scale)

    def l(self, num_output_layers, pyramid_scale):
        stage_infos = [
            ['conv', 24, 3, 1],
            ['conv', 48, 3, 2],
            ['conv', 96, 3, 2],
            ['lcsp', 192, 3, 4],
            ['lcsp', 384, 3, 6],
            ['lcsp', 768, 3, 8],
            ['csp', 768, 3, 4],
        ]
        return self.build_model(stage_infos, num_output_layers, pyramid_scale)

    def x(self, num_output_layers, pyramid_scale):
        stage_infos = [
            ['conv', 32, 3, 1],
            ['conv', 64, 3, 2],
            ['conv', 128, 3, 3],
            ['lcsp', 256, 3, 4],
            ['lcsp', 512, 3, 6],
            ['lcsp', 1024, 3, 8],
            ['csp', 1024, 3, 4],
        ]
        return self.build_model(stage_infos, num_output_layers, pyramid_scale)

    def build_model(self, stage_infos, num_output_layers, pyramid_scale):
        assert len(stage_infos) == 7
        input_layer = tf.keras.layers.Input(shape=self.input_shape, name='sbd_input')
        final_layers = []
        if not self.cfg.p6_model:
            stage_infos.pop(6)
        stages = []
        x = input_layer
        model_p = 6 if self.cfg.p6_model else 5
        for p, stage_info in enumerate(stage_infos):
            if p >= 3:
                x = self.dropout(x)
            name, channels, kernel_size, depth = stage_info
            if name in ['conv', 'lcsp']:
                depth -= 1
            x = self.stage_block(x, name, channels, kernel_size, depth)
            stages.append(x)
            if p < model_p:
                x = self.conv2d(x, channels, 3, strides=2)
        final_layers.append(stages.pop(-1))
        for i, stage_info in enumerate(reversed(stage_infos)):
            p = model_p - i
            if p == model_p:
                continue
            name, channels, kernel_size, depth = stage_info
            x = self.conv2d(x, channels, 1)
            x = self.upsampling2d(x)
            x = self.add([x, stages.pop(-1)])
            x = self.bn(x)
            x = self.stage_block(x, name, channels, kernel_size, depth)
            final_layers.append(x)
            if p == pyramid_scale:
                break
        final_layers = final_layers if num_output_layers == 'm' else [final_layers[-1]]
        output_layers = []
        for i, final_layer in enumerate(final_layers):
            output_layers.append(self.detection_layer(final_layer, f'sbd_output_{i}'))
        return tf.keras.models.Model(input_layer, output_layers if num_output_layers == 'm' else output_layers[0])

    def stage_block(self, x, name, channels, kernel_size, depth):
        available_names = ['conv', 'lcsp', 'csp']
        if name == 'conv':
            for _ in range(depth):
                x = self.conv2d(x, channels, kernel_size)
        elif name == 'lcsp':
            x = self.lcsp_block(x, channels, kernel_size, depth)
        elif name == 'csp':
            x = self.csp_block(x, channels, kernel_size, depth)
        else:
            Logger.error(f'layer block name({name}) is invalid, available_names : {available_names}')
        return x

    def hybrid_attention_block(self, x):
        input_layer = x
        input_filters = input_layer.shape[-1]
        reduced_filters = max(input_filters // 16, 2)
        x = self.conv2d(x, reduced_filters, 1)
        x = self.conv2d(x, reduced_filters, 7)
        x = self.conv2d(x, input_filters, 1, activation='sigmoid')
        return self.multiply([x, input_layer])

    def feature_fusion_block(self, x, layers):
        pool_size = 2
        fusion_layers = [x]
        channels = x.shape[-1]
        for i in range(len(layers)):
            fusion_x = self.maxpooling2d(layers[i], pool_size=pool_size)
            fusion_x = self.conv2d(fusion_x, channels, 1)
            fusion_layers.append(fusion_x)
            pool_size *= 2
        return self.conv2d(self.bn(self.add(fusion_layers)), channels, 1)

    def lcsp_block(self, x, filters, kernel_size, depth):
        half_filters = filters // 2
        x_0 = self.conv2d(x, half_filters, 1)
        x_1 = self.conv2d(x, filters, 1)
        for _ in range(depth):
            x_0 = self.conv2d(x_0, half_filters, kernel_size)
        x_0 = self.conv2d(x_0, filters, 1)
        x = self.add([x_0, x_1])
        x = self.bn(x)
        x = self.conv2d(x, filters, 1)
        return x

    def csp_block(self, x, filters, kernel_size, depth):
        x = self.conv2d(x, x.shape[-1], 1)
        half_filters = filters // 2
        x_half = self.conv2d(x, half_filters, 1)
        x_half_first = x_half
        for _ in range(depth):
            x_half_0 = self.conv2d(x_half, half_filters, kernel_size)
            x_half_1 = self.conv2d(x_half_0, half_filters, kernel_size)
            x_half = self.add([x_half, x_half_1])
        x_half = self.add([x_half, x_half_first])
        x = self.concat([x, x_half])
        x = self.bn(x)
        x = self.conv2d(x, filters, 1)
        return x

    def conv2d(self, x, filters, kernel_size, activation='auto', strides=1, bn=False):
        if activation == 'auto':
            activation = self.cfg.activation
        assert activation in self.available_activations, f'activation must be one of {self.available_activations}'
        is_fused_activation = False if bn else activation in self.fused_activations
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            bias_initializer=self.bias_initializer(),
            activation=activation if is_fused_activation else 'linear',
            padding='same',
            use_bias=not bn,
            kernel_regularizer=self.kernel_regularizer())(x)
        if bn:
            x = self.bn(x)
        if not is_fused_activation:
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
    def maxpooling2d(x, pool_size=2):
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

