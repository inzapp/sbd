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
import numpy as np
import shutil as sh
import tensorflow as tf

from glob import glob
from model import Model
from util import ModelUtil
from time import time, sleep
from box_colors import colors
from lr_scheduler import LRScheduler
from generator import DataGenerator
from mAP_calculator import calc_mean_average_precision
from loss import confidence_loss, confidence_with_bbox_loss, sbd_loss, IGNORED_LOSS


class SBD:
    def __init__(self, cfg_path, training=True):
        config = self.load_cfg(cfg_path)
        self.cfg_path = cfg_path
        self.pretrained_model_path = config['pretrained_model_path']
        input_rows = config['input_rows']
        input_cols = config['input_cols']
        self.input_channels = config['input_channels']
        input_shape = (input_rows, input_cols, self.input_channels)
        train_image_path = config['train_image_path']
        validation_image_path = config['validation_image_path']
        multi_classification_at_same_box = config['multi_classification_at_same_box']
        ignore_scale = config['ignore_scale']
        aug_scale = config['aug_scale']
        aug_brightness = config['aug_brightness']
        aug_contrast = config['aug_contrast']
        batch_size = config['batch_size']
        self.class_names_file_path = config['class_names_file_path']
        self.lr = config['lr']
        self.obj_alpha_param = config['obj_alpha'] 
        self.obj_gamma_param = config['obj_gamma']
        self.cls_alpha_param = config['cls_alpha'] 
        self.cls_gamma_param = config['cls_gamma']
        self.obj_alphas = None
        self.obj_gammas = None
        self.cls_alphas = None
        self.cls_gammas = None
        self.l2 = config['l2']
        self.momentum = config['momentum']
        self.label_smoothing = config['smoothing']
        self.warm_up = config['warm_up']
        self.decay_step = config['decay_step']
        self.iterations = config['iterations']
        self.optimizer = config['optimizer']
        self.lr_policy = config['lr_policy']
        self.model_name = config['model_name']
        self.model_type = config['model_type']
        self.training_view = config['training_view']
        self.map_checkpoint_interval = config['map_checkpoint_interval']
        self.live_view_previous_time = time()
        self.checkpoint_path = self.new_checkpoint_path()
        self.pretrained_iteration_count = 0
        self.best_mean_ap = 0.0

        self.class_names, self.num_classes = ModelUtil.init_class_names(self.class_names_file_path)
        self.use_pretrained_model = False
        if self.pretrained_model_path.endswith('.h5') and training:
            self.load_model(self.pretrained_model_path)
            self.use_pretrained_model = True

        if not self.use_pretrained_model:
            if self.num_classes == 0:
                ModelUtil.print_error_exit(f'classes file not found. file path : {self.class_names_file_path}')
            if self.optimizer == 'adam':
                self.l2 = 0.0
            self.model = Model(input_shape=input_shape, output_channel=self.num_classes + 5, l2=self.l2).build(self.model_type)

        if type(self.model.output_shape) == tuple:
            self.num_output_layers = 1
        else:
            self.num_output_layers = len(self.model.output_shape)

        self.train_image_paths = ModelUtil.init_image_paths(train_image_path)
        self.validation_image_paths = ModelUtil.init_image_paths(validation_image_path)

        self.train_data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            output_shape=self.model.output_shape,
            batch_size=batch_size,
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_scale=ignore_scale,
            aug_scale=aug_scale,
            aug_brightness=aug_brightness,
            aug_contrast=aug_contrast)
        self.validation_data_generator = DataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=input_shape,
            output_shape=self.model.output_shape,
            batch_size=batch_size,
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_scale=ignore_scale,
            aug_scale=aug_scale,
            aug_brightness=aug_brightness,
            aug_contrast=aug_contrast)
        self.train_data_generator_for_check = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            output_shape=self.model.output_shape,
            batch_size=ModelUtil.get_zero_mod_batch_size(len(self.train_image_paths)),
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_scale=ignore_scale,
            aug_scale=1.0,
            aug_brightness=aug_brightness,
            aug_contrast=aug_contrast)
        self.validation_data_generator_for_check = DataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=input_shape,
            output_shape=self.model.output_shape,
            batch_size=ModelUtil.get_zero_mod_batch_size(len(self.validation_image_paths)),
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_scale=ignore_scale,
            aug_scale=1.0,
            aug_brightness=aug_brightness,
            aug_contrast=aug_contrast)
        np.set_printoptions(precision=6)

    def load_cfg(self, cfg_path):
        if not os.path.exists(cfg_path):
            ModelUtil.print_error_exit(f'invalid cfg path. file not found : {cfg_path}')
        if not os.path.isfile(cfg_path):
            ModelUtil.print_error_exit(f'invalid file format, is not file : {cfg_path}')
        with open(cfg_path, 'rt') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def make_checkpoint_dir(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def init_checkpoint_dir(self):
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

    def load_model(self, model_path):
        if os.path.exists(model_path) and os.path.isfile(model_path):
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(model_path)
        else:
            ModelUtil.print_error_exit(f'pretrained model not found. model path : {model_path}')

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

    def new_checkpoint_path(self):
        inc = 0
        base_path = f'checkpoint/{self.model_name}'
        while True:
            checkpoint_path = f'{base_path}/m{inc}'
            if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
                inc += 1
            else:
                break
        return checkpoint_path

    def get_optimizer(self, optimizer_str):
        lr = self.lr if self.lr_policy == 'constant' else 0.0
        if optimizer_str == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=self.momentum, nesterov=True)
        elif optimizer_str == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=self.momentum)
        elif optimizer_str == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            print(f'\n\nunknown optimizer : {optimizer_str}')
            optimizer = None
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
                        ModelUtil.print_error_exit(f'invalid alpha or gamma value value : {val}')
            else:
                ModelUtil.print_error_exit(f'list length of alpha is must be equal with models output layer count {num_output_layers}')
        else:
            ModelUtil.print_error_exit(f'invalid type ({param_type}) of alpha. type must be float or list')
        return params

    def set_alpha_gamma(self):
        num_output_layers = len(self.model.output_shape) if type(self.model.output_shape) is list else 1
        self.obj_alphas = self.convert_alpha_gamma_to_list(self.obj_alpha_param, num_output_layers)
        self.obj_gammas = self.convert_alpha_gamma_to_list(self.obj_gamma_param, num_output_layers)
        self.cls_alphas = self.convert_alpha_gamma_to_list(self.cls_alpha_param, num_output_layers)
        self.cls_gammas = self.convert_alpha_gamma_to_list(self.cls_gamma_param, num_output_layers)

    def fit(self):
        self.init_checkpoint_dir()
        gflops = ModelUtil.get_gflops(self.model)
        self.save_last_model(iteration_count=self.pretrained_iteration_count)
        self.model.summary()
        print(f'\nGFLOPs : {gflops:.4f}')
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print(f'validate on {len(self.validation_image_paths)} samples.')

        print('\nlabel check in train data...')
        self.train_data_generator_for_check.check_label()
        print('\nlabel check in validation data...')
        self.validation_data_generator_for_check.check_label()
        print('\ncalculate virtual anchor...')
        self.train_data_generator.calculate_virtual_anchor()
        print('\ncalculate BPR(Best Possible Recall)...')
        self.train_data_generator.calculate_best_possible_recall()
        self.set_alpha_gamma()
        print('\nstart test forward for checking forwarding time.')
        if ModelUtil.available_device() == 'gpu':
            ModelUtil.check_forwarding_time(self.model, device='gpu')
        ModelUtil.check_forwarding_time(self.model, device='cpu')
        if self.use_pretrained_model:
            print(f'\nstart training with pretrained model : {self.pretrained_model_path}')
        else:
            print('\nstart training')
        self.train()

    def compute_gradient(self, model, optimizer, loss_function, x, y_true, mask, num_output_layers, obj_alphas, obj_gammas, cls_alphas, cls_gammas, label_smoothing):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            if num_output_layers == 1:
                confidence_loss, bbox_loss, classification_loss = loss_function(y_true, y_pred, mask, obj_alphas[0], obj_gammas[0], cls_alphas[0], cls_gammas[0], label_smoothing)
            else:
                confidence_loss, bbox_loss, classification_loss = 0.0, 0.0, 0.0
                for i in range(num_output_layers):
                    _confidence_loss, _bbox_loss, _classification_loss = loss_function(y_true[i], y_pred[i], mask[i], obj_alphas[i], obj_gammas[i], cls_alphas[i], cls_gammas[i], label_smoothing)
                    confidence_loss += _confidence_loss
                    bbox_loss += _bbox_loss
                    classification_loss += _classification_loss
            loss = confidence_loss
            if bbox_loss != IGNORED_LOSS:
                loss += bbox_loss
            if classification_loss != IGNORED_LOSS:
                loss += classification_loss
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return confidence_loss, bbox_loss, classification_loss

    def refresh_model_and_optimizer(self, model, optimizer_str):
        sleep(0.2)
        model_path = f'model.h5'
        model.save(model_path, include_optimizer=False)
        sleep(0.2)
        model = tf.keras.models.load_model(model_path, compile=False)
        optimizer = self.get_optimizer(optimizer_str)
        return model, optimizer

    def build_loss_str(self, iteration_count, loss_vars):
        confidence_loss, bbox_loss, classification_loss = loss_vars
        loss_str = f'\r[iteration_count : {iteration_count:6d}]'
        loss_str += f' confidence_loss : {confidence_loss:>8.4f}'
        if bbox_loss != IGNORED_LOSS:
            loss_str += f', bbox_loss: {bbox_loss:>8.4f}'
        if classification_loss != IGNORED_LOSS:
            loss_str += f', classification_loss : {classification_loss:>8.4f}'
        return loss_str

    def train(self):
        iteration_count = self.pretrained_iteration_count
        compute_gradient_tf = tf.function(self.compute_gradient)
        self.model, optimizer = self.refresh_model_and_optimizer(self.model, self.optimizer)
        lr_scheduler = LRScheduler(iterations=self.iterations, lr=self.lr, warm_up=self.warm_up, policy=self.lr_policy, decay_step=self.decay_step)
        print(f'model will be save to {self.checkpoint_path}')
        while True:
            for _ in range(len(self.train_data_generator)):
                batch_x, batch_y, mask = self.train_data_generator.load()
                lr_scheduler.update(optimizer, iteration_count)
                loss_vars = compute_gradient_tf(
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
                    self.label_smoothing)
                iteration_count += 1
                print(self.build_loss_str(iteration_count, loss_vars), end='')
                warm_up_end = iteration_count >= int(self.iterations * self.warm_up)
                if iteration_count % 2000 == 0:
                    self.save_last_model(iteration_count=iteration_count)
                if warm_up_end:
                    if self.training_view:
                        self.training_view_function()
                    if self.map_checkpoint_interval > 0 and iteration_count % self.map_checkpoint_interval == 0 and iteration_count < self.iterations:
                        self.save_model_with_map()
                if iteration_count == self.iterations:
                    self.save_model_with_map()
                    self.remove_last_model()
                    print('\n\ntrain end successfully')
                    return

    @staticmethod
    @tf.function
    def decode_bounding_box(output_tensor, confidence_threshold):
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

        xmin = cx - (w * 0.5)
        ymin = cy - (h * 0.5)
        xmax = cx + (w * 0.5)
        ymax = cy + (h * 0.5)

        confidence = tf.expand_dims(confidence, axis=-1)
        xmin = tf.expand_dims(xmin, axis=-1)
        ymin = tf.expand_dims(ymin, axis=-1)
        xmax = tf.expand_dims(xmax, axis=-1)
        ymax = tf.expand_dims(ymax, axis=-1)
        max_class_index = tf.expand_dims(max_class_index, axis=-1)
        result_tensor = tf.concat([confidence, ymin, xmin, ymax, xmax, max_class_index], axis=-1)  # box format must be [ymin, xmin, ymax, xmax] for using tf.image.non_max_suppression()
        boxes_before_nms = tf.gather_nd(result_tensor, over_confidence_indices)
        return boxes_before_nms

    @staticmethod
    @tf.function(experimental_relax_shapes=True)
    def tf_nms(boxes, nms_iou_threshold, num_classes, max_box_size_per_class):
        scores = boxes[:, 0]
        boxes = boxes[:, 1:]

        selected_indices = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        for class_index in tf.range(num_classes):
            class_indices = tf.where(tf.equal(tf.cast(boxes[:, 4], dtype=tf.int32), class_index))
            class_indices = tf.cast(class_indices, dtype=tf.int32)  # cast to int32
            class_boxes = tf.gather_nd(boxes[:, :4], class_indices)
            class_scores = tf.gather_nd(scores, class_indices)
            nms_indices = tf.image.non_max_suppression(class_boxes, class_scores, max_output_size=max_box_size_per_class, iou_threshold=nms_iou_threshold)
            selected_indices = selected_indices.write(class_index, tf.gather(class_indices, nms_indices))
        selected_indices = selected_indices.concat()

        selected_boxes = tf.gather(boxes, selected_indices)
        selected_boxes = tf.reshape(selected_boxes, (-1, 5))
        selected_scores = tf.gather(scores, selected_indices)

        boxes_after_nms = tf.concat([selected_scores, selected_boxes], axis=1)
        return boxes_after_nms

    @staticmethod
    def predict(model, img, device, confidence_threshold=0.2, nms_iou_threshold=0.45, verbose=False):
        """
        Detect object in image using trained SBD model.
        :param model: model for for forward image.
        :param img: (width, height, channel) formatted image to be predicted.
        :param device: cpu or gpu device.
        :param confidence_threshold: threshold confidence score to detect object.
        :param nms_iou_threshold: threshold to remove overlapped detection.
        :param verbose: print detected box info if True.
        :return: bounding boxes dictionary array sorted by x position.
        each dictionary has class index and bbox info: [x1, y1, x2, y2].
        """
        input_shape = model.input_shape[1:]
        input_width, input_height, _ = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        output_shape = model.output_shape
        num_output_layers = 1 if type(output_shape) == tuple else len(output_shape)

        img = ModelUtil.resize(img, (input_width, input_height))
        x = np.reshape(ModelUtil.preprocess(img), (1,) + input_shape)
        y = ModelUtil.graph_forward(model, x, device)
        y = np.array(y)
        if num_output_layers == 1:
            y = [y]

        boxes_before_nms_list = []
        for layer_index in range(num_output_layers):
            output_tensor = y[layer_index][0]
            boxes_before_nms_list += list(SBD.decode_bounding_box(output_tensor, confidence_threshold).numpy())
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
        boxes = ModelUtil.nms(boxes_before_nms_dicts, nms_iou_threshold)
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
        return boxes

    def bounding_box(self, img, boxes, font_scale=0.4, show_class_with_score=True):
        """
        draw bounding bbox using result of SBD.predict function.
        :param img: image to be predicted.
        :param boxes: result value of SBD.predict() function.
        :param font_scale: scale of font.
        :param show_class_with_score: draw bounding box with class and score label if True, else draw bounding box only
        :return: image of bounding boxed.
        """
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
            label_font_color = (0, 0, 0) if ModelUtil.is_background_color_bright(label_background_color) else (255, 255, 255)
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

    def predict_video(self, video_path, confidence_threshold=0.2, device='cpu', show_class_with_score=True):
        """
        Equal to the evaluate function. video path is required.
        """
        if not (os.path.exists(video_path) and os.path.isfile(video_path)):
            ModelUtil.print_error_exit(f'video not found. video path : {video_path}')
        cap = cv2.VideoCapture(video_path)
        while True:
            frame_exist, raw = cap.read()
            if not frame_exist:
                print('frame not exists')
                break
            x = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if self.model.input.shape[-1] == 1 else raw.copy()
            boxes = SBD.predict(self.model, x, device=device, confidence_threshold=confidence_threshold)
            boxed_image = self.bounding_box(raw, boxes, show_class_with_score=show_class_with_score)
            cv2.imshow('video', boxed_image)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
        cap.release()
        cv2.destroyAllWindows()

    def predict_images(self, dataset='validation', confidence_threshold=0.2, device='cpu', show_class_with_score=True):
        """
        Equal to the evaluate function. image paths are required.
        """
        input_width, input_height, input_channel = ModelUtil.get_width_height_channel_from_input_shape(self.model.input_shape[1:])
        if dataset == 'train':
            image_paths = self.train_image_paths
        elif dataset == 'validation':
            image_paths = self.validation_image_paths
        else:
            print(f'invalid dataset : [{dataset}]')
            return
        for path in image_paths:
            print(f'image path : {path}')
            raw, raw_bgr, _ = ModelUtil.load_img(path, input_channel)
            boxes = SBD.predict(self.model, raw, device=device, verbose=True, confidence_threshold=confidence_threshold)
            raw_bgr = cv2.resize(raw_bgr, (input_width, input_height), interpolation=cv2.INTER_AREA)
            boxed_image = self.bounding_box(raw_bgr, boxes, show_class_with_score=show_class_with_score)
            cv2.imshow('res', boxed_image)
            key = cv2.waitKey(0)
            if key == 27:
                break

    def calculate_map(self, dataset, device, confidence_threshold, tp_iou_threshold, cached):
        assert dataset in ['train', 'validation']
        image_paths = self.train_image_paths if dataset == 'train' else self.validation_image_paths
        device = ModelUtil.available_device() if device == 'auto' else device
        return calc_mean_average_precision(
            model=self.model,
            image_paths=image_paths,
            device=device,
            confidence_threshold=confidence_threshold,
            tp_iou_threshold=tp_iou_threshold,
            classes_txt_path=self.class_names_file_path,
            cached=cached)

    def remove_last_model(self):
        for last_model_path in glob(f'{self.checkpoint_path}/last_*_iter.h5'):
            os.remove(last_model_path)

    def save_last_model(self, iteration_count):
        self.make_checkpoint_dir()
        save_path = f'{self.checkpoint_path}/last_{iteration_count}_iter.h5'
        self.model.save(save_path, include_optimizer=False)
        tmp_path = f'{save_path}.tmp'
        sh.move(save_path, tmp_path)
        self.remove_last_model()
        sh.move(tmp_path, save_path)
        return save_path

    def save_model_with_map(self, dataset='validation', device='auto', confidence_threshold=0.2, tp_iou_threshold=0.5, cached=False):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        mean_ap, f1_score, iou, tp, fp, fn, confidence, txt_content = self.calculate_map(
            dataset=dataset,
            device=device,
            confidence_threshold=confidence_threshold,
            tp_iou_threshold=tp_iou_threshold,
            cached=cached)
        new_best_model = False
        if mean_ap >= self.best_mean_ap:
            self.best_mean_ap = mean_ap
            new_best_model = True
        if new_best_model:
            best_model_path = f'{self.checkpoint_path}/best.h5'
            self.make_checkpoint_dir()
            self.model.save(best_model_path, include_optimizer=False)
            with open(f'{self.checkpoint_path}/map.txt', 'wt') as f:
                f.write(txt_content)
            print(f'new best model saved to [{best_model_path}]')

    def training_view_function(self):
        """
        During training, the image is forwarded in real time, showing the results are shown.
        """
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            if np.random.uniform() > 0.5:
                img_path = np.random.choice(self.train_image_paths)
            else:
                img_path = np.random.choice(self.validation_image_paths)
            img, raw_bgr, _ = ModelUtil.load_img(img_path, self.input_channels)
            boxes = SBD.predict(self.model, img, device='cpu')
            boxed_image = self.bounding_box(raw_bgr, boxes)
            cv2.imshow('training view', boxed_image)
            cv2.waitKey(1)

