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
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KMP_AFFINITY'] = 'noverbose'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NCCL_P2P_DISABLE'] = '1'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import cv2
import yaml
import threading
import numpy as np
import shutil as sh
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from model import Model
from logger import Logger
from loss import sbd_loss
from eta import ETACalculator
from box_colors import colors
from keras_flops import get_flops
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ckpt_manager import CheckpointManager
from time import time, sleep, perf_counter
from map_boxes import mean_average_precision_for_boxes
from concurrent.futures.thread import ThreadPoolExecutor


class TrainingConfig:
    def __init__(self, cfg_path):
        self.__d = self.load(cfg_path)
        self.sync_attribute()

    def sync_attribute(self):
        for key, value in self.__d.items():
            setattr(self, key, value)

    def __get_value_from_yaml(self, cfg, key, default, parse_type, required):
        try:
            value = parse_type(cfg[key])
            if parse_type is str and value.lower() in ['none', 'null']:
                value = None
            return value
        except:
            if required:
                Logger.error(f'cfg parse failure, {key} is required')
            return default

    def set_config(self, key, value):
        self.__d[key] = value
        setattr(self, key, value)

    def load(self, cfg_path):
        cfg = None
        if not (os.path.exists(cfg_path) and os.path.isfile(cfg_path)):
            Logger.error(f'cfg not found, path : {cfg_path}')

        with open(cfg_path, 'rt') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        d = {}
        d['devices'] = self.__get_value_from_yaml(cfg, 'devices', [0], list, required=False)
        d['pretrained_model_path'] = self.__get_value_from_yaml(cfg, 'pretrained_model_path', None, str, required=False)
        d['train_data_path'] = self.__get_value_from_yaml(cfg, 'train_data_path', None, str, required=True)
        d['validation_data_path'] = self.__get_value_from_yaml(cfg, 'validation_data_path', None, str, required=True)
        d['class_names_file_path'] = self.__get_value_from_yaml(cfg, 'class_names_file_path', None, str, required=True)
        d['input_rows'] = self.__get_value_from_yaml(cfg, 'input_rows', None, int, required=True)
        d['input_cols'] = self.__get_value_from_yaml(cfg, 'input_cols', None, int, required=True)
        d['input_channels'] = self.__get_value_from_yaml(cfg, 'input_channels', None, int, required=True)
        d['model_name'] = self.__get_value_from_yaml(cfg, 'model_name', 'model', str, required=False)
        d['model_type'] = self.__get_value_from_yaml(cfg, 'model_type', None, str, required=True)
        d['activation'] = self.__get_value_from_yaml(cfg, 'activation', 'leaky', str, required=False)
        d['p6_model'] = self.__get_value_from_yaml(cfg, 'p6_model', False, bool, required=False)
        d['optimizer'] = self.__get_value_from_yaml(cfg, 'optimizer', 'sgd', str, required=False)
        d['lr_policy'] = self.__get_value_from_yaml(cfg, 'lr_policy', 'step', str, required=False)
        d['lr'] = self.__get_value_from_yaml(cfg, 'lr', 0.001, float, required=False)
        d['lrf'] = self.__get_value_from_yaml(cfg, 'lrf', 0.05, float, required=False)
        d['l2'] = self.__get_value_from_yaml(cfg, 'l2', 0.0005, float, required=False)
        d['dropout'] = self.__get_value_from_yaml(cfg, 'dropout', 0.0, float, required=False)
        d['obj_target'] = self.__get_value_from_yaml(cfg, 'obj_target', 'iou', str, required=False)
        d['cls_balance'] = self.__get_value_from_yaml(cfg, 'cls_balance', 0.0, float, required=False)
        d['box_weight'] = self.__get_value_from_yaml(cfg, 'box_weight', 1.0, float, required=False)
        d['aug_noise'] = self.__get_value_from_yaml(cfg, 'aug_noise', 0.0, float, required=False)
        d['aug_scale'] = self.__get_value_from_yaml(cfg, 'aug_scale', 0.5, float, required=False)
        d['aug_mixup'] = self.__get_value_from_yaml(cfg, 'aug_mixup', 0.0, float, required=False)
        d['aug_mosaic'] = self.__get_value_from_yaml(cfg, 'aug_mosaic', 0.2, float, required=False)
        d['aug_h_flip'] = self.__get_value_from_yaml(cfg, 'aug_h_flip', False, bool, required=False)
        d['aug_v_flip'] = self.__get_value_from_yaml(cfg, 'aug_v_flip', False, bool, required=False)
        d['aug_contrast'] = self.__get_value_from_yaml(cfg, 'aug_contrast', 0.3, float, required=False)
        d['aug_brightness'] = self.__get_value_from_yaml(cfg, 'aug_brightness', 0.3, float, required=False)
        d['aug_snowstorm'] = self.__get_value_from_yaml(cfg, 'aug_snowstorm', 0.0, float, required=False)
        warm_up = self.__get_value_from_yaml(cfg, 'warm_up', 1000, float, required=False)
        d['warm_up'] = float(warm_up) if 0.0 <= warm_up <= 1.0 else int(warm_up)
        d['momentum'] = self.__get_value_from_yaml(cfg, 'momentum', 0.9, float, required=False)
        d['smoothing'] = self.__get_value_from_yaml(cfg, 'smoothing', 0.0, float, required=False)
        d['heatmap_scale'] = self.__get_value_from_yaml(cfg, 'heatmap_scale', 0.0, float, required=False)
        d['va_iou_threshold'] = self.__get_value_from_yaml(cfg, 'va_iou_threshold', 0.0, float, required=False)
        d['batch_size'] = self.__get_value_from_yaml(cfg, 'batch_size', 4, int, required=False)
        d['max_q_size'] = self.__get_value_from_yaml(cfg, 'max_q_size', 1024, int, required=False)
        d['iterations'] = self.__get_value_from_yaml(cfg, 'iterations', None, int, required=True)
        d['checkpoint_interval'] = self.__get_value_from_yaml(cfg, 'checkpoint_interval', 0, int, required=False)
        d['show_progress'] = self.__get_value_from_yaml(cfg, 'show_progress', False, bool, required=False)
        d['treat_unknown_as_class'] = self.__get_value_from_yaml(cfg, 'treat_unknown_as_class', False, bool, required=False)
        d['multi_classification_at_same_box'] = self.__get_value_from_yaml(cfg, 'multi_classification_at_same_box', False, bool, required=False)
        d['fix_seed'] = self.__get_value_from_yaml(cfg, 'fix_seed', False, bool, required=False)
        return d

    def save(self, cfg_path):
        with open(cfg_path, 'wt') as f:
            yaml.dump(self.__d, f, default_flow_style=False, sort_keys=False)

    def print_cfg(self):
        print(self.__d)


class SBD(CheckpointManager):
    def __init__(self, cfg):
        super().__init__()
        if cfg.p6_model:
            assert cfg.input_rows % 64 == 0 and cfg.input_cols % 64 == 0, 'input_rows, input_cols of p6 model must be multiple of 64'
        else:
            assert cfg.input_rows % 32 == 0 and cfg.input_cols % 32 == 0, 'input_rows, input_cols must be multiple of 32'
        assert cfg.input_channels in [1, 3], 'input_channels must be in [1, 3]'
        assert cfg.max_q_size >= cfg.batch_size
        assert cfg.obj_target in ['binary', 'iou']
        self.cfg = cfg

        if self.cfg.obj_target == 'iou' and self.cfg.heatmap_scale > 0.0:
            Logger.warn(f'heatmap_scale({self.cfg.heatmap_scale}) will be ignored with iou obj_target')

        if self.cfg.checkpoint_interval == 0:
            self.cfg.checkpoint_interval = self.cfg.iterations

        if self.cfg.fix_seed:
            self.set_global_seed()

        self.show_progress_previous_time = time()

        is_train_data_path_valid = True
        if self.cfg.train_data_path.endswith('.txt'):
            if not self.is_path_valid(self.cfg.train_data_path, path_type='file'):
                is_train_data_path_valid = False
        else:
            if not self.is_path_valid(self.cfg.train_data_path, path_type='dir'):
                is_train_data_path_valid = False
        if not is_train_data_path_valid:
            Logger.error(f'train data path is not valid : {self.cfg.train_data_path}')

        is_validation_data_path_valid = True
        if self.cfg.validation_data_path.endswith('.txt'):
            if not self.is_path_valid(self.cfg.validation_data_path, path_type='file'):
                is_validation_data_path_valid = False
        else:
            if not self.is_path_valid(self.cfg.validation_data_path, path_type='dir'):
                is_validation_data_path_valid = False
        if not is_validation_data_path_valid:
            Logger.error(f'validation data path is not valid : {self.cfg.validation_data_path}')

        self.strategy, self.primary_context = self.get_context(self.cfg.devices)
        self.optimizer = self.get_optimizer(self.strategy, self.cfg.optimizer, self.cfg.lr, self.cfg.momentum, self.cfg.lr_policy)

        if not self.is_path_valid(self.cfg.class_names_file_path, path_type='file'):
            Logger.error(f'class_names_file_path is not valid : {self.cfg.class_names_file_path}')

        self.class_names, self.num_classes, self.unknown_class_index = self.get_class_infos(self.cfg.class_names_file_path)

        if self.cfg.pretrained_model_path is None:
            self.model = Model(cfg=self.cfg, num_classes=self.num_classes).build(self.strategy, self.optimizer, self.cfg.model_type)
        else:
            self.model = self.load_model(self.cfg.pretrained_model_path, self.strategy, self.optimizer)
            Logger.info(f'load model success => {self.cfg.pretrained_model_path}')

        self.pool = ThreadPoolExecutor(8)

        if type(self.model.output_shape) == tuple:
            self.num_output_layers = 1
        else:
            self.num_output_layers = len(self.model.output_shape)

        self.train_data_generator = DataGenerator(
            cfg=self.cfg,
            output_shape=self.model.output_shape,
            class_names=self.class_names,
            unknown_class_index=self.unknown_class_index,
            training=True)
        self.validation_data_generator = DataGenerator(
            cfg=self.cfg,
            output_shape=self.model.output_shape,
            class_names=self.class_names,
            unknown_class_index=self.unknown_class_index)

        self.last_annotations_csv_path = None
        self.last_predictions_csv_path = None
        self.best_annotations_csv_path = None
        self.best_predictions_csv_path = None

    def set_global_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)
        Logger.info(f'global seed fixed to {seed}')

    def is_path_valid(self, path, path_type):
        assert path_type in ['file', 'dir']
        if path_type == 'file':
            return (path is not None) and os.path.exists(path) and os.path.isfile(path)
        else:
            return (path is not None) and os.path.exists(path) and os.path.isdir(path)

    def get_context(self, user_devices):
        strategy = None
        primary_context = None
        tf.keras.backend.clear_session()
        if len(user_devices) == 0:
            tf.config.set_visible_devices([], 'GPU')
            primary_context = tf.device('/cpu:0')
            strategy = tf.distribute.get_strategy()
        else:
            tf.config.set_soft_device_placement(True)
            physical_devices = tf.config.list_physical_devices('GPU')

            available_device_indices = list(map(int, [int(device.name.split(':')[-1]) for device in physical_devices]))

            visible_devices = []
            for user_device_index in user_devices:
                if user_device_index not in available_device_indices:
                    Logger.error(f'invalid device index {user_device_index}. available device indices : {available_device_indices}')
                else:
                    visible_devices.append(physical_devices[user_device_index])
            tf.config.set_visible_devices(visible_devices, 'GPU')

            primary_device = user_devices[0]
            primary_context = tf.device(f'/gpu:{primary_device}')
            if len(user_devices) == 1:
                strategy = tf.distribute.get_strategy()
            else:
                strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{i}' for i in user_devices])
        return strategy, primary_context

    def get_optimizer(self, strategy, optimizer_str, lr, momentum, lr_policy):
        available_optimizer_strs = ['sgd', 'adam']
        optimizer_str = optimizer_str.lower()
        assert optimizer_str in available_optimizer_strs, f'invalid optimizer {optimizer_str}, available optimizers : {available_optimizer_strs}'
        lr = lr if lr_policy == 'constant' else 0.0
        with strategy.scope():
            if optimizer_str == 'sgd':
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True)
            elif optimizer_str == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=momentum)
                self.cfg.set_config('l2', 0.0)
        return optimizer

    def load_model(self, path, strategy, optimizer):
        if path == 'auto':
            auto_model_path = None
            if auto_model_path is None:
                auto_model_path = self.get_best_model_path(path='.')
            if auto_model_path is None:
                auto_model_path = self.get_last_model_path(path='.')
            if auto_model_path is not None:
                self.cfg.set_config('pretrained_model_path', auto_model_path)
                path = auto_model_path

        if not self.is_path_valid(path, path_type='file'):
            Logger.error(f'model not found : {self.cfg.pretrained_model_path}')

        with strategy.scope():
            model = tf.keras.models.load_model(path, compile=False, custom_objects={'tf': tf})
            model.compile(optimizer=optimizer)
        input_shape = model.input_shape[1:]
        self.cfg.set_config('input_rows', input_shape[0])
        self.cfg.set_config('input_cols', input_shape[1])
        self.cfg.set_config('input_channels', input_shape[2])
        self.pretrained_iteration_count = self.parse_pretrained_iteration_count(path)
        return model

    def get_class_infos(self, class_names_file_path):
        class_names = []
        num_classes = 0
        unknown_class_index = -1
        with open(class_names_file_path, 'rt') as classes_file:
            class_names = [s.replace('\n', '') for s in classes_file.readlines()]
            if not self.cfg.treat_unknown_as_class:
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

    def init_checkpoint_dir_extra(self):
        self.cfg.save(f'{self.checkpoint_path}/cfg.yaml')
        sh.copy(self.cfg.class_names_file_path, f'{self.checkpoint_path}/classes.txt')

    def check_forwarding_time(self, model, context, name):
        input_shape = model.input_shape[1:]
        mul = 1
        for val in input_shape:
            mul *= val

        forward_count = 32
        noise = np.random.uniform(0.0, 1.0, mul * forward_count)
        noise = np.asarray(noise).reshape((forward_count, 1) + input_shape).astype(np.float32)
        SBD.graph_forward(model, noise[0], context)  # only first forward is slow, skip first forward in check forwarding time

        st = perf_counter()
        for i in range(forward_count):
            SBD.graph_forward(model, noise[i], context)
        et = perf_counter()
        forwarding_time = ((et - st) / forward_count) * 1000.0
        Logger.info(f'model forwarding time with {name} : {forwarding_time:.2f} ms')

    @tf.function
    def compute_gradient(self, args):
        _, _, model, optimizer, loss_function, x, y_true, extra, iou_obj_target, num_output_layers, box_weight, label_smoothing = args
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            obj_loss, obj_pos_loss, obj_neg_loss, box_loss, cls_loss, num_pos, num_neg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if num_output_layers == 1:
                y_true = [y_true]
                y_pred = [y_pred]
                extra = [extra]
            for i in range(num_output_layers):
                _obj_pos_loss, _obj_neg_loss, _num_pos, _num_neg, _box_loss, _cls_loss = loss_function(
                    y_true[i], y_pred[i], extra[i], iou_obj_target, box_weight, label_smoothing)
                num_pos += _num_pos
                num_neg += _num_neg
                if _num_pos > 0.0:
                    obj_pos_loss += _obj_pos_loss
                    box_loss += _box_loss
                    cls_loss += _cls_loss
                if _num_neg > 0.0:
                    obj_neg_loss += _obj_neg_loss
            if num_pos > 0.0:
                obj_loss += obj_pos_loss / num_pos
                box_loss /= num_pos
                cls_loss /= num_pos
            if num_neg > 0.0:
                obj_loss += obj_neg_loss * tf.sqrt(1.0 / num_neg)
            loss = obj_loss + box_loss + cls_loss
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return obj_loss, box_loss, cls_loss

    @tf.function
    def compute_gradient_distributed(self, args):
        strategy, train_step, *_ = args
        obj_loss, box_loss, cls_loss = strategy.run(train_step, args=(args,))
        obj_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, obj_loss, axis=None)
        box_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, box_loss, axis=None)
        cls_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, cls_loss, axis=None)
        return obj_loss, box_loss, cls_loss

    def build_loss_str(self, progress_str, loss_vars):
        obj_loss, box_loss, cls_loss = loss_vars
        if np.isnan(obj_loss) or np.isnan(box_loss) or np.isnan(cls_loss):
            print()
            Logger.error(f'training exited caused by nan loss => obj_loss({obj_loss:.4f}), box_loss({box_loss:.4f}), cls_loss({cls_loss:.4f}), force exited',
                callback=self.train_data_generator.exit)
        loss_str = f'\r{progress_str}'
        loss_str += f' obj_loss : {obj_loss:>8.4f}'
        loss_str += f', box_loss : {box_loss:>8.4f}'
        loss_str += f', cls_loss : {cls_loss:>8.4f}'
        return loss_str

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

    def make_predictions_csv(self, model, image_paths, context, csv_path):
        fs = []
        input_channel = model.input_shape[1:][-1]
        for path in image_paths:
            fs.append(self.pool.submit(self.train_data_generator.load_image, path))
        csv = 'ImageID,LabelName,Conf,XMin,XMax,YMin,YMax\n'
        for f in tqdm(fs, desc='predictions csv creation'):
            img, path = f.result()
            _, boxes = self.predict(model, img, confidence_threshold=0.001, context=context)
            csv += self.convert_boxes_to_csv_lines(path, boxes)
        with open(csv_path, 'wt') as f:
            f.writelines(csv)

    def save_best_model_extra_data(self, txt_content, best_confidence_thresholds=None):
        with open(f'{self.checkpoint_path}/map.txt', 'wt') as f:
            f.write(txt_content)
        if self.is_path_valid(self.last_annotations_csv_path, path_type='file'):
            sh.copy(self.last_annotations_csv_path, self.best_annotations_csv_path)
        if self.is_path_valid(self.last_predictions_csv_path, path_type='file'):
            sh.copy(self.last_predictions_csv_path, self.best_predictions_csv_path)
        if best_confidence_thresholds:
            threshold_content = ''
            for i, threshold in enumerate(best_confidence_thresholds):
                if i == 0:
                    threshold_content += f'{threshold:.2f}'
                else:
                    threshold_content += f', {threshold:.2f}'
            threshold_content += '\n'
            best_thresholds_path = f'{self.checkpoint_path}/thresholds.txt'
            with open(best_thresholds_path, 'wt') as f:
                f.write(threshold_content)
            Logger.info(f'best f1 score class confidence thresholds saved to {best_thresholds_path}')

    def remove_last_extra_data(self):
        os.remove(self.last_annotations_csv_path)
        os.remove(self.last_predictions_csv_path)

    def evaluate(self,
            dataset='validation',
            cached=False,
            confidence_threshold=0.2,
            tp_iou_threshold=0.5,
            annotations_csv_path='',
            predictions_csv_path='',
            find_best_threshold=False,
            verbose=True):
        assert dataset in ['train', 'validation']
        if annotations_csv_path == '':
            annotations_csv_path = self.last_annotations_csv_path
        if predictions_csv_path == '':
            predictions_csv_path = self.last_predictions_csv_path

        if not cached:
            image_paths = self.train_data_generator.data_paths if dataset == 'train' else self.validation_data_generator.data_paths
            self.make_annotations_csv(image_paths, self.unknown_class_index, annotations_csv_path)
            self.make_predictions_csv(self.model, image_paths, self.primary_context, predictions_csv_path)

        mean_ap, txt_content, best_thresholds = mean_average_precision_for_boxes(
            ann=annotations_csv_path,
            pred=predictions_csv_path,
            confidence_threshold_for_f1=confidence_threshold,
            iou_threshold=tp_iou_threshold,
            classes_txt_path=self.cfg.class_names_file_path,
            find_best_threshold=find_best_threshold,
            verbose=verbose)
        return mean_ap, txt_content, best_thresholds

    def save_best_thresholds(self):
        Logger.info('searching best f1 score class confidence thresholds...')
        _, txt_content, best_confidence_thresholds = self.evaluate(
            cached=True,
            verbose=False,
            annotations_csv_path=self.best_annotations_csv_path,
            predictions_csv_path=self.best_predictions_csv_path,
            find_best_threshold=True)
        self.save_best_model_extra_data(txt_content, best_confidence_thresholds=best_confidence_thresholds)

    def is_background_color_bright(self, bgr):
        tmp = np.zeros((1, 1), dtype=np.uint8)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tmp, (0, 0), (1, 1), bgr, -1)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        return tmp[0][0] > 127

    def draw_box(self, img, boxes, font_scale=0.4, show_class=True):
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
            if show_class:
                cv2.rectangle(img, (x1 - 1, y1 - bh), (x1 - 1 + bw, y1), label_background_color, -1)
                cv2.putText(img, label_text, (x1 + padding - 1, y1 - baseline - padding), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
        return img

    def show_progress(self):
        cur_time = time()
        if cur_time - self.show_progress_previous_time > 0.5:
            self.show_progress_previous_time = cur_time
            if np.random.uniform() > 0.5:
                img_path = np.random.choice(self.train_data_generator.data_paths)
            else:
                img_path = np.random.choice(self.validation_data_generator.data_paths)
            img, _ = self.train_data_generator.load_image(img_path)
            img, boxes = self.predict(self.model, img, context=self.primary_context, heatmap=True)
            img = self.draw_box(img, boxes)
            cv2.imshow('progress', img)
            key = cv2.waitKey(1)
            if key == 27:
                self.cfg.show_progress = False
                cv2.destroyAllWindows()

    @tf.function
    def decode_bounding_box(self, output_tensor, confidence_threshold):
        output_shape = tf.shape(output_tensor)
        rows, cols = output_shape[0], output_shape[1]
        rows_f = tf.cast(rows, dtype=tf.float32)
        cols_f = tf.cast(cols, dtype=tf.float32)

        confidence = output_tensor[:, :, 0]
        max_class_score = tf.reduce_max(output_tensor[:, :, 5:], axis=-1)
        max_class_index = tf.cast(tf.argmax(output_tensor[:, :, 5:], axis=-1), dtype=tf.float32)
        confidence *= max_class_score
        over_confidence_indices = tf.where(confidence > confidence_threshold)

        cx = output_tensor[:, :, 1]
        cy = output_tensor[:, :, 2]
        w = output_tensor[:, :, 3]
        h = output_tensor[:, :, 4]

        x_grid, y_grid = tf.meshgrid(tf.range(cols_f), tf.range(rows_f), indexing='xy')

        cx = (x_grid + cx) / cols_f
        cy = (y_grid + cy) / rows_f

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
        result_tensor = tf.concat([confidence, xmin, ymin, xmax, ymax, max_class_index], axis=-1)
        boxes_before_nms = tf.gather_nd(result_tensor, over_confidence_indices)
        return boxes_before_nms

    @staticmethod
    @tf.function
    def graph_forward(model, x, context):
        with context:
            return model(x, training=False)

    def nms(self, boxes, nms_iou_threshold=0.45):
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        for i in range(len(boxes) - 1):
            if boxes[i]['discard']:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j]['discard'] or boxes[i]['class'] != boxes[j]['class']:
                    continue
                if self.train_data_generator.iou(boxes[i]['bbox_norm'], boxes[j]['bbox_norm']) > nms_iou_threshold:
                    boxes[j]['discard'] = True

        y_pred_copy = np.asarray(boxes.copy())
        boxes = []
        for i in range(len(y_pred_copy)):
            if not y_pred_copy[i]['discard']:
                boxes.append(y_pred_copy[i])
        return boxes

    def predict(self, model, img, context, confidence_threshold=0.2, verbose=False, heatmap=True):
        input_shape = model.input_shape[1:]
        self.cfg.input_rows, self.cfg.input_cols = input_shape[:2]
        output_shape = model.output_shape
        num_output_layers = 1 if type(output_shape) == tuple else len(output_shape)

        img_resized = self.train_data_generator.resize(img, (self.cfg.input_cols, self.cfg.input_rows))
        x = self.train_data_generator.preprocess(img_resized, batch_axis=True)
        y = SBD.graph_forward(model, x, context)
        if num_output_layers == 1:
            y = [y]

        if type(confidence_threshold) is list:
            confidence_thresholds = confidence_threshold
        else:
            confidence_thresholds = [confidence_threshold for _ in range(self.num_classes)]
        confidence_threshold_min = min(confidence_thresholds)

        proposals = []
        for layer_index in range(num_output_layers):
            output_tensor = y[layer_index][0]
            proposals += list(self.decode_bounding_box(output_tensor, confidence_threshold_min).numpy())

        proposal_dicts = []
        for box in proposals:
            confidence = float(box[0])
            x1, y1, x2, y2 = np.clip(np.array(list(map(float, box[1:5]))), 0.0, 1.0)
            class_index = int(box[5])
            if confidence > confidence_thresholds[class_index]:
                proposal_dicts.append({
                    'confidence': confidence,
                    'bbox_norm': [x1, y1, x2, y2],
                    'class': class_index,
                    'discard': False})

        boxes = self.nms(proposal_dicts)
        if verbose:
            print(f'before nms box count : {len(proposal_dicts)}')
            print(f'after  nms box count : {len(boxes)}')
            for box_info in boxes:
                class_index = box_info['class']
                confidence = box_info['confidence']
                x1, y1, x2, y2 = box_info['bbox_norm']
                print(f'confidence({confidence:.4f}), bbox({x1:.6f}, {y1:.6f}, {x2:.6f}, {y2:.6f}), class({self.class_names[class_index]})')
            print()

        if heatmap:
            if num_output_layers == 1:
                objectness = y[0][:, :, :, 0][0]
                img = self.train_data_generator.blend_heatmap(img, objectness)

        return img, boxes

    def read_video_frame_into_q(self, video_path, frame_queue, read_flag_list, thread_end_flag_list, lock):
        cap = cv2.VideoCapture(video_path)
        while read_flag_list[0]:
            frame_exist, bgr = cap.read()
            if not frame_exist:
                break
            with lock:
                if len(frame_queue) == 0:
                    frame_queue.append(bgr)
                else:
                    frame_queue[0] = bgr
            sleep(0)
        cap.release()
        Logger.info('receive thread VideoCapture release success')
        thread_end_flag_list[0] = True

    def load_best_thresholds(self, thresholds_path):
        with open(thresholds_path, 'rt') as f:
            line = f.readlines()[0]
        best_confidence_thresholds = list(map(float, line.split(',')))
        best_confidence_thresholds_len = len(best_confidence_thresholds)
        if best_confidence_thresholds_len != self.num_classes:
            Logger.error(f'best_confidence_thresholds length({best_confidence_thresholds_len} is not matched with num_classes({self.num_classes}))')
        info_content = [f'best confidence threshold load success => {thresholds_path}']
        for i, class_name in enumerate(self.class_names):
            info_content.append(f'{class_name} : {best_confidence_thresholds[i]:.2f}')
        Logger.info(info_content)
        return best_confidence_thresholds

    def detect(self, path='', dataset='validation', confidence_threshold=0.2, tp_iou_threshold=0.5, show_class=True, width=0, height=0, heatmap=False, thresholds_path='', save_label=False):
        image_paths = []
        if path == '':
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths = self.train_data_generator.data_paths
            elif dataset == 'validation':
                image_paths = self.validation_data_generator.data_paths
            if len(image_paths) == 0:
                Logger.error('no images found')
            detect_type = 'image'
        else:
            if path.endswith('.mp4'):
                if not self.is_path_valid(path, path_type='file'):
                    Logger.error(f'file not found : {path}')
                detect_type = 'video'
            elif path.startswith('rtsp://'):
                detect_type = 'rtsp'
            else:
                if not os.path.exists(path):
                    Logger.error(f'path not exists : {path}')
                if os.path.isfile(path):
                    if path.endswith('.jpg'):
                        image_paths = [path]
                    else:
                        Logger.error('invalid extension. jpg is available extension only')
                elif os.path.isdir(path):
                    image_paths = glob(f'{path}/**/*.jpg', recursive=True)
                else:
                    Logger.error(f'invalid file format : [{path}]')
                detect_type = 'image'

        view_width, view_height = 0, 0
        if width > 0 and height > 0:
            view_width, view_height = width, height
        else:
            input_height, input_width, _ = self.model.input_shape[1:]
            view_width, view_height = input_width, input_height

        if thresholds_path != '':
            if self.is_path_valid(thresholds_path, path_type='file'):
                confidence_threshold = self.load_best_thresholds(thresholds_path)
            else:
                Logger.info(f'{thresholds_path} not found. confidence threshold will be {confidence_threshold:.2f}')

        assert detect_type in ['image', 'video', 'rtsp']
        if detect_type == 'image':
            for path in image_paths:
                print(f'image path : {path}')
                img, _ = self.train_data_generator.load_image(path)
                img, boxes = self.predict(self.model, img, context=self.primary_context, verbose=True, confidence_threshold=confidence_threshold, heatmap=heatmap)
                img = self.train_data_generator.resize(img, (view_width, view_height))
                img = self.draw_box(img, boxes, show_class=show_class)
                cv2.imshow('res', img)
                key = cv2.waitKey(0)
                if key == 27:
                    break
        elif detect_type == 'video':
            cap = cv2.VideoCapture(path)
            while True:
                frame_exist, img_bgr = cap.read()
                if not frame_exist:
                    Logger.info('frame not exists')
                    break
                img, boxes = self.predict(self.model, img_bgr, context=self.primary_context, confidence_threshold=confidence_threshold, heatmap=heatmap)
                img = self.train_data_generator.resize(img, (view_width, view_height))
                img = self.draw_box(img, boxes, show_class=show_class)
                cv2.imshow('video', img)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            rtsp_url = path
            lock, frame_queue, read_flag_list, thread_end_flag_list = threading.Lock(), [], [True], [False]
            receive_thread = threading.Thread(target=self.read_video_frame_into_q, args=(rtsp_url, frame_queue, read_flag_list, thread_end_flag_list, lock))
            receive_thread.daemon = True
            receive_thread.start()
            while True:
                if thread_end_flag_list[0]:
                    break
                img_bgr = None
                with lock:
                    if frame_queue:
                        img_bgr = frame_queue[0].copy()
                if img_bgr is None:
                    Logger.info(f'wait for receive thread initializing...')
                    sleep(1.0)
                    continue

                img, boxes = self.predict(self.model, img_bgr, context=self.primary_context, confidence_threshold=confidence_threshold, heatmap=heatmap)
                img = self.train_data_generator.resize(img, (view_width, view_height))
                img = self.draw_box(img, boxes, show_class=show_class)
                cv2.imshow('rtsp', img)
                key = cv2.waitKey(1)
                if key == 27:
                    read_flag_list[0] = False
                    while not thread_end_flag_list[0]:
                        Logger.info('wait for receive thread end...')
                        sleep(1.0)
                    break
            cv2.destroyAllWindows()

    def auto_label(self, image_path, confidence_threshold, thresholds_path):
        input_shape = self.model.input_shape[1:]
        channel = input_shape[-1]

        if thresholds_path != '':
            if self.is_path_valid(thresholds_path, path_type='file'):
                confidence_threshold = self.load_best_thresholds(thresholds_path)
            else:
                Logger.info(f'{thresholds_path} not found. confidence threshold will be {confidence_threshold:.2f}')

        image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        try:
            sh.copy(self.cfg.class_names_file_path, f'{image_path}/classes.txt')
        except sh.SameFileError:
            pass

        fs = []
        for path in image_paths:
            fs.append(self.pool.submit(self.is_path_valid, self.train_data_generator.label_path(path), 'file'))
        label_file_count = 0
        for f in fs:
            if f.result():
                label_file_count += 1
        if label_file_count > 0:
            ans = input(f'{label_file_count} label files will be overwritten. continue? [Y/n] : ')
            if ans not in ['y', 'Y']:
                Logger.info('canceled')
                return

        fs = []
        for path in image_paths:
            fs.append(self.pool.submit(self.train_data_generator.load_image, path))

        for f in tqdm(fs):
            img, path = f.result()
            _, boxes = self.predict(self.model, img, self.primary_context, confidence_threshold=confidence_threshold)
            boxes = sorted(boxes, key=lambda x: ((x['bbox_norm'][2] - x['bbox_norm'][0]) * (x['bbox_norm'][3] - x['bbox_norm'][1])), reverse=True)  # sort by area desc
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
            with open(self.train_data_generator.label_path(path), 'wt') as f_label:
                f_label.write(label_content)

    def train(self):
        if self.pretrained_iteration_count >= self.cfg.iterations:
            Logger.error(f'pretrained iteration count {self.pretrained_iteration_count} is greater or equal than target iterations {self.cfg.iterations}')

        self.train_data_generator.check_label()
        self.validation_data_generator.check_label()
        self.train_data_generator.calculate_virtual_anchor()
        # self.train_data_generator.calculate_best_possible_recall()
        Logger.info('start test forward for checking forwarding time')
        self.check_forwarding_time(self.model, context=self.primary_context, name='gpu' if len(self.cfg.devices) > 0 else 'cpu')

        print()
        self.cfg.print_cfg()
        print()

        Logger.info(f'input_shape : {self.model.input_shape}')
        Logger.info(f'output_shape : {self.model.output_shape}\n')
        Logger.info(f'model_type : {self.cfg.model_type}')
        Logger.info(f'parameters : {self.model.count_params():,}\n')
        Logger.info(f'train on {len(self.train_data_generator.data_paths)} samples')
        Logger.info(f'validate on {len(self.validation_data_generator.data_paths)} samples\n')

        self.train_data_generator.start()
        if self.cfg.pretrained_model_path is not None:
            Logger.info(f'start training with pretrained model : {self.cfg.pretrained_model_path}')
        else:
            Logger.info('start training')

        self.init_checkpoint_dir(model_name=self.cfg.model_name, model_type=self.cfg.model_type, extra_function=self.init_checkpoint_dir_extra)
        self.last_annotations_csv_path = f'{self.checkpoint_path}/.annotations.csv'
        self.last_predictions_csv_path = f'{self.checkpoint_path}/.predictions.csv'
        self.best_annotations_csv_path = f'{self.checkpoint_path}/annotations.csv'
        self.best_predictions_csv_path = f'{self.checkpoint_path}/predictions.csv'

        iou_obj_target = float(self.cfg.obj_target == 'iou')
        iteration_count = self.pretrained_iteration_count
        if len(self.cfg.devices) <= 1:
            train_step = self.compute_gradient
        else:
            train_step = self.compute_gradient_distributed
        lr_scheduler = LRScheduler(iterations=self.cfg.iterations, lr=self.cfg.lr, lrf=self.cfg.lrf, warm_up=self.cfg.warm_up, policy=self.cfg.lr_policy)
        eta_calculator = ETACalculator(iterations=self.cfg.iterations, start_iteration=iteration_count)
        eta_calculator.start()
        Logger.info(f'model will be save to {self.checkpoint_path}')
        while True:
            batch_x, batch_y, batch_extra = self.train_data_generator.load()
            lr_scheduler.update(self.optimizer, iteration_count)
            loss_vars = train_step((
                self.strategy,
                self.compute_gradient,
                self.model,
                self.optimizer,
                sbd_loss,
                batch_x,
                batch_y,
                batch_extra,
                iou_obj_target,
                self.num_output_layers,
                self.cfg.box_weight,
                self.cfg.smoothing))

            iteration_count += 1
            print(self.build_loss_str(eta_calculator.update(iteration_count), loss_vars), end='')
            warm_up_end = iteration_count >= lr_scheduler.warm_up_iterations

            if iteration_count % 2000 == 0:
                self.save_last_model(self.model, iteration_count=iteration_count)
            if warm_up_end:
                if self.cfg.show_progress:
                    self.show_progress()
                if iteration_count % self.cfg.checkpoint_interval == 0:
                    self.train_data_generator.pause()
                    mean_ap, txt_content, _ = self.evaluate()
                    best_model_path = self.save_best_model(self.model, iteration_count, metric=mean_ap, mode='max', content=f'_mAP_{mean_ap:.4f}')
                    if best_model_path:
                        self.save_best_model_extra_data(txt_content=txt_content)
                        Logger.info(f'[{iteration_count} iter] evaluation success with mAP {mean_ap:.4f}, new best model is saved to {best_model_path}\n')
                    else:
                        Logger.info(f'[{iteration_count} iter] evaluation success with mAP {mean_ap:.4f}\n')
                    self.train_data_generator.resume()
            if iteration_count == self.cfg.iterations:
                self.train_data_generator.stop()
                self.save_best_thresholds()
                self.remove_last_model()
                self.remove_last_extra_data()
                Logger.info('training end successfully')
                return

