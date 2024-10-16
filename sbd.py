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
import os
import cv2
import yaml
import warnings
import numpy as np
import shutil as sh
import silence_tensorflow.auto
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from model import Model
from logger import Logger
from eta import ETACalculator
from box_colors import colors
from keras_flops import get_flops
from generator import DataGenerator
from lr_scheduler import LRScheduler
from loss import sbd_loss, IGNORED_LOSS
from time import time, sleep, perf_counter
from map_boxes import mean_average_precision_for_boxes
from concurrent.futures.thread import ThreadPoolExecutor


class SBD:
    def __init__(self, cfg_path, training=True):
        config = self.load_cfg(cfg_path)
        self.cfg_path = cfg_path
        self.devices = config['devices']
        self.pretrained_model_path = config['pretrained_model_path']
        input_rows = config['input_rows']
        input_cols = config['input_cols']
        self.input_channels = config['input_channels']
        train_image_path = config['train_image_path']
        validation_image_path = config['validation_image_path']
        multi_classification_at_same_box = config['multi_classification_at_same_box']
        ignore_scale = config['ignore_scale']
        virtual_anchor_iou_threshold = config['va_iou_threshold']
        aug_noise = config['aug_noise']
        aug_scale = config['aug_scale']
        aug_mosaic = config['aug_mosaic']
        aug_h_flip = config['aug_h_flip']
        aug_v_flip = config['aug_v_flip']
        aug_contrast = config['aug_contrast']
        aug_brightness = config['aug_brightness']
        aug_snowstorm = config['aug_snowstorm']
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        max_q_size = config['max_q_size']
        self.class_names_file_path = config['class_names_file_path']
        self.lr = config['lr']
        self.lrf = config['lrf']
        self.obj_alpha_param = config['obj_alpha']
        self.obj_gamma_param = config['obj_gamma']
        self.cls_alpha_param = config['cls_alpha']
        self.cls_gamma_param = config['cls_gamma']
        self.box_weight = config['box_weight']
        self.obj_alphas = None
        self.obj_gammas = None
        self.cls_alphas = None
        self.cls_gammas = None
        self.l2 = config['l2']
        self.drop_rate = config['dropout']
        self.momentum = config['momentum']
        self.label_smoothing = config['smoothing']
        self.warm_up = config['warm_up']
        self.iterations = config['iterations']
        self.optimizer = config['optimizer'].lower()
        self.lr_policy = config['lr_policy']
        self.model_name = config['model_name']
        self.model_type = config['model_type']
        self.activation = config['activation']
        self.p6_model = config['p6_model']
        self.training_view = config['training_view']
        self.treat_unknown_as_class = config['treat_unknown_as_class']
        self.map_checkpoint_interval = config['map_checkpoint_interval']
        self.live_view_previous_time = time()
        self.checkpoint_path = None
        self.annotations_csv_path_last = None
        self.predictions_csv_path_last = None
        self.annotations_csv_path_best = None
        self.predictions_csv_path_best = None
        self.pretrained_iteration_count = 0
        self.best_mean_ap = 0.0
        warnings.filterwarnings(action='ignore')

        self.use_pretrained_model = False
        self.model = None
        self.class_names, self.num_classes, self.unknown_class_index = self.init_class_names(self.class_names_file_path)
        self.pool = ThreadPoolExecutor(8)

        assert type(self.devices) is list
        self.strategy = None
        tf.keras.backend.clear_session()
        tf.config.set_soft_device_placement(True)
        physical_devices = tf.config.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
        visible_devices = []
        if len(self.devices) == 0:
            self.primary_device = '/cpu:0'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        else:
            available_gpu_indexes = list(map(int, [gpu.name[-1] for gpu in physical_devices]))
            for device_index in self.devices:
                if device_index not in available_gpu_indexes:
                    Logger.error(f'invalid gpu index {device_index}. available gpu index : {available_gpu_indexes}')
                for physical_device in physical_devices:
                    if int(physical_device.name[-1]) == device_index:
                        visible_devices.append(physical_device)
            if len(self.devices) > 1:
                self.strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{i}' for i in self.devices])
            self.primary_device = f'/gpu:{self.devices[0]}'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['NCCL_P2P_DISABLE'] = '1'
        tf.config.set_visible_devices(visible_devices, 'GPU')

        if self.pretrained_model_path.endswith('.h5') and training:
            self.load_model(self.pretrained_model_path)
            self.use_pretrained_model = True

        if self.p6_model:
            assert input_rows % 64 == 0 and input_cols % 64 == 0, 'input_rows, input_cols of p6 model must be multiple of 64'
        else:
            assert input_rows % 32 == 0 and input_cols % 32 == 0, 'input_rows, input_cols must be multiple of 32'
        assert self.input_channels in [1, 3]
        input_shape = (input_rows, input_cols, self.input_channels)
        if not self.use_pretrained_model:
            if self.num_classes == 0:
                Logger.error(f'classes file not found. file path : {self.class_names_file_path}')
            if self.optimizer == 'adam':
                self.l2 = 0.0
            with self.device_context():
                self.model = Model(
                    input_shape=input_shape,
                    output_channel=self.num_classes + 5,
                    p6=self.p6_model,
                    l2=self.l2,
                    drop_rate=self.drop_rate,
                    activation=self.activation).build(self.model_type)

        if type(self.model.output_shape) == tuple:
            if self.model_type[1] == 'm':
                new_type = f'{self.model_type[0]}1{self.model_type[2:]}'
                msg = f'{self.model_type} model with pyramid scale {self.model_type[3]} is same with {new_type}.'
                msg += f' use {new_type} instead to to clarify that model is one output layer'
                Logger.error(msg)
            self.num_output_layers = 1
        else:
            self.num_output_layers = len(self.model.output_shape)

        self.train_image_paths = self.init_image_paths(train_image_path)
        self.validation_image_paths = self.init_image_paths(validation_image_path)

        self.data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            output_shape=self.model.output_shape,
            batch_size=batch_size,
            num_workers=num_workers,
            max_q_size=max_q_size,
            unknown_class_index=self.unknown_class_index,
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_scale=ignore_scale,
            virtual_anchor_iou_threshold=virtual_anchor_iou_threshold,
            aug_noise=aug_noise,
            aug_scale=aug_scale,
            aug_mosaic=aug_mosaic,
            aug_h_flip=aug_h_flip,
            aug_v_flip=aug_v_flip,
            aug_contrast=aug_contrast,
            aug_brightness=aug_brightness,
            aug_snowstorm=aug_snowstorm,
            primary_device=self.primary_device)
        np.set_printoptions(precision=6)

    def is_file_exists(self, path):
        return os.path.exists(path) and os.path.isfile(path)

    def init_class_names(self, class_names_file_path):
        class_names = []
        num_classes = 0
        unknown_class_index = -1
        if self.is_file_exists(class_names_file_path):
            with open(class_names_file_path, 'rt') as classes_file:
                class_names = [s.replace('\n', '') for s in classes_file.readlines()]
                if not self.treat_unknown_as_class:
                    for i, class_name in enumerate(class_names):
                        if class_name == 'unknown':
                            if unknown_class_index == -1:
                                unknown_class_index = i
                            else:
                                Logger.error(f'unknown class count in {class_names_file_path} must be 1')
                num_classes = len(class_names)
                if unknown_class_index > -1:
                    num_classes -= 1
                if num_classes <= 0:
                    Logger.error('cannot build model with unknown class only')
        return class_names, num_classes, unknown_class_index

    def init_image_paths(self, image_path):
        if image_path.endswith('.txt'):
            with open(image_path, 'rt') as f:
                image_paths = f.readlines()
            for i in range(len(image_paths)):
                image_paths[i] = image_paths[i].replace('\n', '')
        else:
            image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        return image_paths

    def load_cfg(self, cfg_path):
        if not self.is_file_exists(cfg_path):
            Logger.error(f'invalid cfg path. file not found : {cfg_path}')
        with open(cfg_path, 'rt') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def make_checkpoint_dir(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def init_checkpoint_dir(self):
        inc = 0
        while True:
            new_checkpoint_path = f'checkpoint/{self.model_name}/{self.model_type.lower()}_{inc}'
            if os.path.exists(new_checkpoint_path) and os.path.isdir(new_checkpoint_path):
                inc += 1
            else:
                break

        self.checkpoint_path = new_checkpoint_path
        self.make_checkpoint_dir()
        cfg_content = ''
        with open(self.cfg_path, 'rt') as f:
            lines = f.readlines()
        for line in lines:
            if line.strip().startswith('#') or line.strip().replace('\n', '') == '':
                continue
            cfg_content += line
        with open(f'{self.checkpoint_path}/cfg.yaml', 'wt') as f:
            f.writelines(cfg_content)
        sh.copy(self.class_names_file_path, self.checkpoint_path)

        self.annotations_csv_path_last = f'{self.checkpoint_path}/annotations_last.csv'
        self.predictions_csv_path_last = f'{self.checkpoint_path}/predictions_last.csv'
        self.annotations_csv_path_best = f'{self.checkpoint_path}/annotations.csv'
        self.predictions_csv_path_best = f'{self.checkpoint_path}/predictions.csv'

    def device_context(self):
        return tf.device(self.primary_device) if self.strategy is None else self.strategy.scope()

    def load_model_with_device(self, model_path):
        model = None
        with self.device_context():
            model = tf.keras.models.load_model(model_path, compile=False)
        return model

    def load_model(self, model_path):
        if self.is_file_exists(model_path):
            self.model = self.load_model_with_device(model_path)
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(model_path)
        else:
            Logger.error(f'pretrained model not found. model path : {model_path}')

    def parse_pretrained_iteration_count(self, pretrained_model_path):
        iteration_count = 0
        sp = f'{os.path.basename(pretrained_model_path)[:-3]}'.split('_')
        for i in range(len(sp)):
            if sp[i] == 'iter' and i > 0:
                try:
                    iteration_count = int(sp[i-1])
                except:
                    pass
                break
        return iteration_count

    def get_optimizer(self, optimizer_str):
        lr = self.lr if self.lr_policy == 'constant' else 0.0
        if optimizer_str == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=self.momentum, nesterov=True)
        elif optimizer_str == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=self.momentum)
        elif optimizer_str == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            Logger.error(f'unknown optimizer : {optimizer_str}, possible optimizers [sgd, adam, rmsprop]')
        return optimizer

    def convert_alpha_gamma_to_list(self, param, num_output_layers):
        params = None
        param_type = type(param)
        if param_type is float:
            params = [param for _ in range(num_output_layers)]
        elif param_type is list:
            if len(param) == num_output_layers:
                params = param
                for val in params:
                    try:
                        float(val)
                    except:
                        Logger.error(f'invalid alpha or gamma value value : {val}')
            else:
                Logger.error(f'list length of alpha is must be equal with models output layer count {num_output_layers}')
        else:
            Logger.error(f'invalid type ({param_type}) of alpha. type must be float or list')
        return params

    def set_alpha_gamma(self):
        num_output_layers = len(self.model.output_shape) if type(self.model.output_shape) is list else 1
        self.obj_alphas = self.convert_alpha_gamma_to_list(self.obj_alpha_param, num_output_layers)
        self.obj_gammas = self.convert_alpha_gamma_to_list(self.obj_gamma_param, num_output_layers)
        self.cls_alphas = self.convert_alpha_gamma_to_list(self.cls_alpha_param, num_output_layers)
        self.cls_gammas = self.convert_alpha_gamma_to_list(self.cls_gamma_param, num_output_layers)

    def check_forwarding_time(self, model, device):
        input_shape = model.input_shape[1:]
        mul = 1
        for val in input_shape:
            mul *= val

        forward_count = 32
        noise = np.random.uniform(0.0, 1.0, mul * forward_count)
        noise = np.asarray(noise).reshape((forward_count, 1) + input_shape).astype('float32')
        SBD.graph_forward(model, noise[0], device)  # only first forward is slow, skip first forward in check forwarding time

        st = perf_counter()
        for i in range(forward_count):
            SBD.graph_forward(model, noise[i], device)
        et = perf_counter()
        forwarding_time = ((et - st) / forward_count) * 1000.0
        Logger.info(f'model forwarding time with {device[1:4]} : {forwarding_time:.2f} ms')

    def compute_gradient(self, args):
        _strategy, _train_step, model, optimizer, loss_function, x, y_true, mask, num_output_layers, obj_alphas, obj_gammas, cls_alphas, cls_gammas, box_weight, label_smoothing = args
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            obj_loss, box_loss, cls_loss = 0.0, 0.0, 0.0
            if num_output_layers == 1:
                obj_loss, box_loss, cls_loss = loss_function(
                    y_true, y_pred, mask, obj_alphas[0], obj_gammas[0], cls_alphas[0], cls_gammas[0], box_weight, label_smoothing)
            else:
                for i in range(num_output_layers):
                    _obj_loss, _box_loss, _cls_loss = loss_function(
                         y_true[i], y_pred[i], mask[i], obj_alphas[i], obj_gammas[i], cls_alphas[i], cls_gammas[i], box_weight, label_smoothing)
                    obj_loss += _obj_loss
                    box_loss = box_loss + _box_loss if _box_loss != IGNORED_LOSS else IGNORED_LOSS
                    cls_loss = cls_loss + _cls_loss if _cls_loss != IGNORED_LOSS else IGNORED_LOSS
            loss = obj_loss
            if box_loss != IGNORED_LOSS:
                loss += box_loss
            if cls_loss != IGNORED_LOSS:
                loss += cls_loss
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return obj_loss, box_loss, cls_loss

    def distributed_train_step(self, args):
        strategy, train_step, *_ = args
        obj_loss, box_loss, cls_loss = strategy.run(train_step, args=(args,))
        obj_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, obj_loss, axis=None)
        box_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, box_loss, axis=None)
        cls_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, cls_loss, axis=None)
        return obj_loss, box_loss, cls_loss

    def refresh(self, model, optimizer_str):
        sleep(0.2)
        model_path = f'{self.checkpoint_path}/model.h5'
        model.save(model_path, include_optimizer=False)
        sleep(0.2)
        model = self.load_model_with_device(model_path)
        optimizer = self.get_optimizer(optimizer_str)
        model.compile(optimizer=optimizer)
        os.remove(model_path)
        return model, optimizer

    def build_loss_str(self, progress_str, loss_vars):
        obj_loss, box_loss, cls_loss = loss_vars
        loss_str = f'\r{progress_str}'
        loss_str += f' obj_loss : {obj_loss:>8.4f}'
        if box_loss != IGNORED_LOSS:
            loss_str += f', box_loss : {box_loss:>8.4f}'
        if cls_loss != IGNORED_LOSS:
            loss_str += f', cls_loss : {cls_loss:>8.4f}'
        return loss_str

    def train(self):
        if self.pretrained_iteration_count >= self.iterations:
            Logger.error(f'pretrained iteration count {self.pretrained_iteration_count} is greater or equal than target iterations {self.iterations}')
        with self.device_context():
            self.data_generator.check_label(self.train_image_paths, self.class_names, 'train')
            self.data_generator.check_label(self.validation_image_paths, self.class_names, 'validation')
            self.data_generator.calculate_virtual_anchor()
            # self.data_generator.calculate_best_possible_recall()
            self.set_alpha_gamma()
            Logger.info('start test forward for checking forwarding time.')
            if self.primary_device.find('gpu') > -1:
                self.check_forwarding_time(self.model, device=self.primary_device)
            self.check_forwarding_time(self.model, device='/cpu:0')
            print()
            Logger.info(f'input_shape : {self.model.input_shape}')
            Logger.info(f'output_shape : {self.model.output_shape}\n')
            Logger.info(f'model_type : {self.model_type}')
            Logger.info(f'parameters : {self.model.count_params():,}\n')
            Logger.info(f'train on {len(self.train_image_paths)} samples.')
            Logger.info(f'validate on {len(self.validation_image_paths)} samples.\n')

            self.data_generator.start()
            if self.use_pretrained_model:
                Logger.info(f'start training with pretrained model : {self.pretrained_model_path}')
            else:
                Logger.info('start training')

            self.init_checkpoint_dir()
            iteration_count = self.pretrained_iteration_count
            train_step = tf.function(self.compute_gradient)
            compute_gradient_tf = train_step
            if self.strategy is not None:
                compute_gradient_tf = tf.function(self.distributed_train_step)
            self.model, optimizer = self.refresh(self.model, self.optimizer)
            lr_scheduler = LRScheduler(iterations=self.iterations, lr=self.lr, lrf=self.lrf, warm_up=self.warm_up, policy=self.lr_policy)
            eta_calculator = ETACalculator(iterations=self.iterations, start_iteration=iteration_count)
            eta_calculator.start()
            Logger.info(f'model will be save to {self.checkpoint_path}')
            while True:
                batch_x, batch_y, mask = self.data_generator.load()
                lr_scheduler.update(optimizer, iteration_count)
                loss_vars = compute_gradient_tf((
                    self.strategy,
                    train_step,
                    self.model,
                    optimizer,
                    sbd_loss,
                    batch_x,
                    batch_y,
                    mask,
                    self.num_output_layers,
                    self.obj_alphas,
                    self.obj_gammas,
                    self.cls_alphas,
                    self.cls_gammas,
                    self.box_weight,
                    self.label_smoothing))
                iteration_count += 1
                progress_str = eta_calculator.update(iteration_count)
                print(self.build_loss_str(progress_str, loss_vars), end='')
                warm_up_end = iteration_count >= lr_scheduler.warm_up_iterations
                if iteration_count % 2000 == 0:
                    self.save_last_model(iteration_count=iteration_count)
                if warm_up_end:
                    if self.training_view:
                        self.training_view_function()
                    if self.map_checkpoint_interval > 0 and iteration_count % self.map_checkpoint_interval == 0 and iteration_count < self.iterations:
                        self.data_generator.pause()
                        self.save_model_with_map()
                        self.data_generator.resume()
                if iteration_count == self.iterations:
                    self.data_generator.stop()
                    self.save_model_with_map()
                    self.remove_last_model()
                    Logger.info('train end successfully')
                    return

    @tf.function
    def decode_bounding_box(self, output_tensor, confidence_threshold):
        output_shape = tf.shape(output_tensor)
        rows, cols = output_shape[0], output_shape[1]

        confidence = output_tensor[:, :, 0]
        max_class_score = tf.reduce_max(output_tensor[:, :, 5:], axis=-1)
        max_class_index = tf.cast(tf.argmax(output_tensor[:, :, 5:], axis=-1), dtype=tf.float32)
        confidence *= max_class_score
        over_confidence_indices = tf.where(confidence > confidence_threshold)

        cx = output_tensor[:, :, 1]
        cy = output_tensor[:, :, 2]
        w = output_tensor[:, :, 3]
        h = output_tensor[:, :, 4]

        x_range = tf.range(cols, dtype=tf.float32)
        x_offset = tf.broadcast_to(x_range, shape=tf.shape(cx))

        y_range = tf.range(rows, dtype=tf.float32)
        y_range = tf.reshape(y_range, shape=(rows, 1))
        y_offset = tf.broadcast_to(y_range, shape=tf.shape(cy))

        cx = (x_offset + cx) / tf.cast(cols, dtype=tf.float32)
        cy = (y_offset + cy) / tf.cast(rows, dtype=tf.float32)

        xmin = tf.clip_by_value(cx - (w * 0.5), 0.0, 1.0)
        ymin = tf.clip_by_value(cy - (h * 0.5), 0.0, 1.0)
        xmax = tf.clip_by_value(cx + (w * 0.5), 0.0, 1.0)
        ymax = tf.clip_by_value(cy + (h * 0.5), 0.0, 1.0)

        confidence = tf.expand_dims(confidence, axis=-1)
        xmin = tf.expand_dims(xmin, axis=-1)
        ymin = tf.expand_dims(ymin, axis=-1)
        xmax = tf.expand_dims(xmax, axis=-1)
        ymax = tf.expand_dims(ymax, axis=-1)
        max_class_index = tf.expand_dims(max_class_index, axis=-1)
        result_tensor = tf.concat([confidence, ymin, xmin, ymax, xmax, max_class_index], axis=-1)
        boxes_before_nms = tf.gather_nd(result_tensor, over_confidence_indices)
        return boxes_before_nms

    @staticmethod
    @tf.function
    def graph_forward(model, x, device):
        with tf.device(device):
            return model(x, training=False)

    def nms(self, boxes, nms_iou_threshold=0.45):
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        for i in range(len(boxes) - 1):
            if boxes[i]['discard']:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j]['discard'] or boxes[i]['class'] != boxes[j]['class']:
                    continue
                if self.data_generator.iou(boxes[i]['bbox_norm'], boxes[j]['bbox_norm']) > nms_iou_threshold:
                    boxes[j]['discard'] = True

        y_pred_copy = np.asarray(boxes.copy())
        boxes = []
        for i in range(len(y_pred_copy)):
            if not y_pred_copy[i]['discard']:
                boxes.append(y_pred_copy[i])
        return boxes

    def predict(self, model, img, device, confidence_threshold=0.2, verbose=False, heatmap=True):
        input_shape = model.input_shape[1:]
        input_rows, input_cols = input_shape[:2]
        output_shape = model.output_shape
        num_output_layers = 1 if type(output_shape) == tuple else len(output_shape)

        img_resized = self.data_generator.resize(img, (input_cols, input_rows))
        x = self.data_generator.preprocess(img_resized, batch_axis=True)
        y = SBD.graph_forward(model, x, device)
        if num_output_layers == 1:
            y = [y]

        boxes_before_nms_list = []
        for layer_index in range(num_output_layers):
            output_tensor = y[layer_index][0]
            boxes_before_nms_list += list(self.decode_bounding_box(output_tensor, confidence_threshold).numpy())

        boxes_before_nms_dicts = []
        for box in boxes_before_nms_list:
            confidence = float(box[0])
            y1, x1, y2, x2 = np.clip(np.array(list(map(float, box[1:5]))), 0.0, 1.0)
            class_index = int(box[5])
            boxes_before_nms_dicts.append({
                'confidence': confidence,
                'bbox_norm': [x1, y1, x2, y2],
                'class': class_index,
                'discard': False})

        boxes = self.nms(boxes_before_nms_dicts)
        if verbose:
            print(f'before nms box count : {len(boxes_before_nms_dicts)}')
            print(f'after  nms box count : {len(boxes)}')
            print()
            for box_info in boxes:
                class_index = box_info['class']
                confidence = box_info['confidence']
                bbox_norm = box_info['bbox_norm']
                print(f'class index : {class_index}')
                print(f'confidence : {confidence:.4f}')
                print(f'bbox(normalized) : {np.array(bbox_norm)}')
                print()
            print()

        if heatmap:
            if num_output_layers == 1:
                objectness = y[0][:, :, :, 0][0]
                img = self.data_generator.blend_heatmap(img, objectness)
            else:
                Logger.warn('heatmap is only possible with one output layer model, flag is ignored')

        return img, boxes

    def auto_label(self, image_path, confidence_threshold, cpu, recursive):
        input_shape = self.model.input_shape[1:]
        channel = input_shape[-1]

        if recursive:
            image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        else:
            image_paths = glob(f'{image_path}/*.jpg', recursive=False)
        try:
            sh.copy(self.class_names_file_path, f'{image_path}/classes.txt')
        except sh.SameFileError:
            pass

        fs = []
        for path in image_paths:
            fs.append(self.pool.submit(self.is_file_exists, self.data_generator.label_path(path)))
        label_file_count = 0
        for f in fs:
            if f.result():
                label_file_count += 1
        if label_file_count > 0:
            ans = input(f'{label_file_count} label files will be overwritten. continue? [Y/n] : ')
            if ans not in ['y', 'Y']:
                print('canceled')
                return

        fs = []
        for path in image_paths:
            fs.append(self.pool.submit(self.data_generator.load_image, path))

        device = '/cpu:0' if cpu else self.primary_device
        for f in tqdm(fs):
            img, path = f.result()
            _, boxes = self.predict(self.model, img, device, confidence_threshold=confidence_threshold)
            label_content = ''
            for box in boxes:
                class_index = box['class']
                xmin, ymin, xmax, ymax = box['bbox_norm']
                w = xmax - xmin
                h = ymax - ymin
                cx = xmin + (w * 0.5)
                cy = ymin + (h * 0.5)
                cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
                label_content += f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'
            with open(self.data_generator.label_path(path), 'wt') as f_label:
                f_label.write(label_content)

    def is_background_color_bright(self, bgr):
        tmp = np.zeros((1, 1), dtype=np.uint8)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tmp, (0, 0), (1, 1), bgr, -1)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        return tmp[0][0] > 127

    def draw_box(self, img, boxes, font_scale=0.4, show_class_with_score=True):
        padding = 5
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_height, img_width = img.shape[:2]
        for i, box in enumerate(boxes):
            class_index = int(box['class'])
            if len(self.class_names) == 0:
                class_name = str(class_index)
            else:
                class_name = self.class_names[class_index].replace('\n', '')
            label_background_color = colors[class_index]
            label_font_color = (0, 0, 0) if self.is_background_color_bright(label_background_color) else (255, 255, 255)
            label_text = f'{class_name}({int(box["confidence"] * 100.0)}%)'
            x1, y1, x2, y2 = box['bbox_norm']
            x1 = int(x1 * img_width)
            y1 = int(y1 * img_height)
            x2 = int(x2 * img_width)
            y2 = int(y2 * img_height)
            l_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
            bw, bh = l_size[0] + (padding * 2), l_size[1] + (padding * 2) + baseline
            cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 1)
            if show_class_with_score:
                cv2.rectangle(img, (x1 - 1, y1 - bh), (x1 - 1 + bw, y1), label_background_color, -1)
                cv2.putText(img, label_text, (x1 + padding - 1, y1 - baseline - padding), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
        return img

    def predict_video(self, path, confidence_threshold=0.2, show_class_with_score=True, width=0, height=0, heatmap=False):
        if not path.startswith('rtsp://') and not self.is_file_exists(path):
            Logger.error(f'video not found. video path : {path}')
        cap = cv2.VideoCapture(path)
        input_height, input_width, _ = self.model.input_shape[1:]
        view_width, view_height = 0, 0
        if width > 0 and height > 0:
            view_width, view_height = width, height
        else:
            view_width, view_height = input_width, input_height
        while True:
            frame_exist, img_bgr = cap.read()
            if not frame_exist:
                Logger.info('frame not exists')
                break
            img, boxes = self.predict(self.model, img_bgr, device=self.primary_device, confidence_threshold=confidence_threshold, heatmap=heatmap)
            img = self.data_generator.resize(img, (view_width, view_height))
            img = self.draw_box(img, boxes, show_class_with_score=show_class_with_score)
            cv2.imshow('video', img)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
        cap.release()
        cv2.destroyAllWindows()

    def predict_images(self, dataset='validation', path='', confidence_threshold=0.2, show_class_with_score=True, width=0, height=0, heatmap=False):
        input_height, input_width, input_channel = self.model.input_shape[1:]
        if path != '':
            if not os.path.exists(path):
                Logger.error(f'path not exists : [{path}]')
            if os.path.isfile(path):
                if path.endswith('.jpg'):
                    image_paths = [path]
                else:
                    Logger.error('invalid extension. jpg is available extension only')
            elif os.path.isdir(path):
                image_paths = glob(f'{path}/*.jpg')
            else:
                Logger.error(f'invalid file format : [{path}]')
        else:
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths = self.train_image_paths
            elif dataset == 'validation':
                image_paths = self.validation_image_paths
        if len(image_paths) == 0:
            Logger.error('no image found')

        view_width, view_height = 0, 0
        if width > 0 and height > 0:
            view_width, view_height = width, height
        else:
            view_width, view_height = input_width, input_height
        for path in image_paths:
            print(f'image path : {path}')
            img, _ = self.data_generator.load_image(path)
            img, boxes = self.predict(self.model, img, device=self.primary_device, verbose=True, confidence_threshold=confidence_threshold, heatmap=heatmap)
            img = self.data_generator.resize(img, (view_width, view_height))
            img = self.draw_box(img, boxes, show_class_with_score=show_class_with_score)
            cv2.imshow('res', img)
            key = cv2.waitKey(0)
            if key == 27:
                break

    def load_label_csv(self, image_path, unknown_class_index):
        csv = ''
        label_path = f'{image_path[:-4]}.txt'
        basename = os.path.basename(image_path)
        with open(label_path, 'rt') as f:
            lines = f.readlines()
        for line in lines:
            class_index, cx, cy, w, h = list(map(float, line.split()))
            class_index = int(class_index)
            if class_index == unknown_class_index:
                continue
            xmin = cx - w * 0.5
            ymin = cy - h * 0.5
            xmax = cx + w * 0.5
            ymax = cy + h * 0.5
            xmin, ymin, xmax, ymax = np.clip(np.array([xmin, ymin, xmax, ymax]), 0.0, 1.0)
            csv += f'{basename},{class_index},{xmin:.6f},{xmax:.6f},{ymin:.6f},{ymax:.6f}\n'
        return csv

    def make_annotations_csv(self, image_paths, unknown_class_index, csv_path):
        fs = []
        for path in image_paths:
            fs.append(self.pool.submit(self.load_label_csv, path, unknown_class_index))
        csv = 'ImageID,LabelName,XMin,XMax,YMin,YMax\n'
        for f in tqdm(fs, desc='annotations csv creation'):
            csv += f.result()
        with open(csv_path, 'wt') as f:
            f.writelines(csv)

    def convert_boxes_to_csv_lines(self, path, boxes):
        csv = ''
        for b in boxes:
            basename = os.path.basename(path)
            confidence = b['confidence']
            class_index = b['class']
            xmin, ymin, xmax, ymax = b['bbox_norm']
            csv += f'{basename},{class_index},{confidence:.6f},{xmin:.6f},{xmax:.6f},{ymin:.6f},{ymax:.6f}\n'
        return csv

    def make_predictions_csv(self, model, image_paths, device, csv_path):
        fs = []
        input_channel = model.input_shape[1:][-1]
        for path in image_paths:
            fs.append(self.pool.submit(self.data_generator.load_image, path))
        csv = 'ImageID,LabelName,Conf,XMin,XMax,YMin,YMax\n'
        for f in tqdm(fs, desc='predictions csv creation'):
            img, path = f.result()
            _, boxes = self.predict(model, img, confidence_threshold=0.001, device=device)
            csv += self.convert_boxes_to_csv_lines(path, boxes)
        with open(csv_path, 'wt') as f:
            f.writelines(csv)

    def calc_mean_average_precision(self, model, image_paths, device, unknown_class_index, confidence_threshold, tp_iou_threshold, classes_txt_path, annotations_csv_path, predictions_csv_path, cached, find_best_threshold):
        if not cached:
            self.make_annotations_csv(image_paths, unknown_class_index, annotations_csv_path)
            self.make_predictions_csv(model, image_paths, device, predictions_csv_path)
        return mean_average_precision_for_boxes(
            ann=annotations_csv_path,
            pred=predictions_csv_path,
            confidence_threshold_for_f1=confidence_threshold,
            iou_threshold=tp_iou_threshold,
            classes_txt_path=classes_txt_path,
            find_best_threshold=find_best_threshold,
            verbose=True)

    def evaluate(self, dataset, confidence_threshold, tp_iou_threshold, cached, find_best_threshold=False, annotations_csv_path='', predictions_csv_path=''):
        assert dataset in ['train', 'validation']
        image_paths = self.train_image_paths if dataset == 'train' else self.validation_image_paths
        if annotations_csv_path == '':
            annotations_csv_path = f'{self.checkpoint_path}/annotations_last.csv'
        if predictions_csv_path == '':
            predictions_csv_path = f'{self.checkpoint_path}/predictions_last.csv'
        return self.calc_mean_average_precision(
            model=self.model,
            image_paths=image_paths,
            device=self.primary_device,
            unknown_class_index=self.unknown_class_index,
            confidence_threshold=confidence_threshold,
            tp_iou_threshold=tp_iou_threshold,
            classes_txt_path=self.class_names_file_path,
            annotations_csv_path=annotations_csv_path,
            predictions_csv_path=predictions_csv_path,
            cached=cached,
            find_best_threshold=find_best_threshold)

    def remove_last_model(self):
        for last_model_path in glob(f'{self.checkpoint_path}/last_*_iter.h5'):
            os.remove(last_model_path)
        if self.is_file_exists(self.annotations_csv_path_last):
            os.remove(self.annotations_csv_path_last)
        if self.is_file_exists(self.predictions_csv_path_last):
            os.remove(self.predictions_csv_path_last)

    def save_last_model(self, iteration_count):
        self.make_checkpoint_dir()
        save_path = f'{self.checkpoint_path}/last_{iteration_count}_iter.h5'
        self.model.save(save_path, include_optimizer=False)
        backup_path = f'{save_path}.bak'
        sh.move(save_path, backup_path)
        self.remove_last_model()
        sh.move(backup_path, save_path)
        return save_path

    def save_model_with_map(self, dataset='validation', confidence_threshold=0.2, tp_iou_threshold=0.5, cached=False):
        print()
        self.make_checkpoint_dir()
        mean_ap, f1_score, iou, tp, fp, fn, confidence, txt_content = self.evaluate(
            dataset=dataset,
            confidence_threshold=confidence_threshold,
            tp_iou_threshold=tp_iou_threshold,
            cached=cached)
        new_best_model = False
        if mean_ap >= self.best_mean_ap:
            self.best_mean_ap = mean_ap
            new_best_model = True
        if new_best_model:
            best_model_path = f'{self.checkpoint_path}/best.h5'
            self.model.save(best_model_path, include_optimizer=False)
            with open(f'{self.checkpoint_path}/map.txt', 'wt') as f:
                f.write(txt_content)
            sh.copy(self.annotations_csv_path_last, self.annotations_csv_path_best)
            sh.copy(self.predictions_csv_path_last, self.predictions_csv_path_best)
            Logger.info(f'new best model saved to [{best_model_path}]')
        print()

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            if np.random.uniform() > 0.5:
                img_path = np.random.choice(self.train_image_paths)
            else:
                img_path = np.random.choice(self.validation_image_paths)
            img, _ = self.data_generator.load_image(img_path)
            img, boxes = self.predict(self.model, img, device=self.primary_device, heatmap=True)
            img = self.draw_box(img, boxes)
            cv2.imshow('training view', img)
            cv2.waitKey(1)

