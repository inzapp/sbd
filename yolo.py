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
from time import time, sleep

import numpy as np
import tensorflow as tf
import cv2

from util import ModelUtil
from box_colors import colors
from generator import YoloDataGenerator
from loss import confidence_loss, confidence_with_bbox_loss, yolo_loss
from mAP_calculator import calc_mean_average_precision
from model import Model
from lr_scheduler import LRScheduler


class Yolo:
    def __init__(self, config):
        pretrained_model_path = config['pretrained_model_path']
        input_shape = config['input_shape']
        train_image_path = config['train_image_path']
        validation_image_path = config['validation_image_path']
        class_names_file_path = config['class_names_file_path']
        multi_classification_at_same_box = config['multi_classification_at_same_box']
        batch_size = config['batch_size']
        self.__lr = config['lr']
        self.__alpha_arg = config['alpha'] 
        self.__alphas = None
        self.__gamma_arg = config['gamma']
        self.__gammas = None
        self.__decay = config['decay']
        self.__momentum = config['momentum']
        self.__label_smoothing = config['label_smoothing']
        self.__burn_in = config['burn_in']
        self.__iterations = config['iterations']
        self.__optimizer = config['optimizer']
        self.__lr_policy = config['lr_policy']
        self.__model_name = config['model_name']
        self.__training_view = config['training_view']
        self.__map_checkpoint = config['map_checkpoint']
        self.__curriculum_iterations = config['curriculum_iterations']
        self.__live_view_previous_time = time()
        self.__checkpoints = config['checkpoints']
        self.__cycle_step = 0
        self.__cycle_length = 2500
        self.max_map, self.max_f1, self.max_map_iou_hm, self.max_f1_iou_hm = 0.0, 0.0, 0.0, 0.0

        self.__input_width, self.__input_height, self.__input_channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        ModelUtil.set_channel_order(input_shape)

        self.__class_names, self.__num_classes = ModelUtil.init_class_names(class_names_file_path)

        if pretrained_model_path != '':
            if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
                self.__model = tf.keras.models.load_model(pretrained_model_path, compile=False)
                print(f'success loading pretrained model : [{pretrained_model_path}]')
            else:
                print(f'pretrained model not found : [{pretrained_model_path}]')
                exit(0)
        else:
            if self.__optimizer == 'adam':
                self.__decay = 0.0
            self.__model = Model(input_shape=input_shape, output_channel=self.__num_classes + 5, decay=self.__decay).build()

        if type(self.__model.output_shape) == tuple:
            self.num_output_layers = 1
        else:
            self.num_output_layers = len(self.__model.output_shape)

        self.__train_image_paths = ModelUtil.init_image_paths(train_image_path)
        self.__validation_image_paths = ModelUtil.init_image_paths(validation_image_path)

        self.__train_data_generator = YoloDataGenerator(
            image_paths=self.__train_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=batch_size,
            multi_classification_at_same_box=multi_classification_at_same_box)
        self.__validation_data_generator = YoloDataGenerator(
            image_paths=self.__validation_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=batch_size,
            multi_classification_at_same_box=multi_classification_at_same_box)
        self.__train_data_generator_for_check = YoloDataGenerator(
            image_paths=self.__train_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=ModelUtil.get_zero_mod_batch_size(len(self.__train_image_paths)),
            multi_classification_at_same_box=multi_classification_at_same_box)
        self.__validation_data_generator_for_check = YoloDataGenerator(
            image_paths=self.__validation_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=ModelUtil.get_zero_mod_batch_size(len(self.__validation_image_paths)),
            multi_classification_at_same_box=multi_classification_at_same_box)

        self.__live_loss_plot = None
        os.makedirs(f'{self.__checkpoints}', exist_ok=True)
        np.set_printoptions(precision=6)

    def __get_optimizer(self, optimizer_str):
        lr = self.__lr if self.__lr_policy == 'constant' else 0.0
        if optimizer_str == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=self.__momentum, nesterov=True)
        elif optimizer_str == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=self.__momentum)
        else:
            print(f'\n\nunknown optimizer : {optimizer_str}')
            optimizer = None
        return optimizer

    def __set_alpha_gamma(self):
        num_output_layers = len(self.__model.output_shape) if type(self.__model.output_shape) is list else 1

        alpha_type = type(self.__alpha_arg)
        if alpha_type is float:
            self.__alphas = [self.__alpha_arg for _ in range(num_output_layers)]
        elif alpha_type is list:
            if len(self.__alpha_arg) == num_output_layers:
                self.__alphas = self.__alpha_arg
            else:
                print(f'list length of alpha is must be equal with models output layer count {num_output_layers}')
                return False
        else:
            print(f'invalid type ({alpha_type}) of alpha. type must be float or list')
            return False

        gamma_type = type(self.__gamma_arg)
        if gamma_type is float:
            self.__gammas = [self.__gamma_arg for _ in range(num_output_layers)]
        elif gamma_type is list:
            if len(self.__gamma_arg) == num_output_layers:
                self.__gammas = self.__gamma_arg
            else:
                print(f'list length of gamma is must be equal with models output layer count {num_output_layers}')
                return False
        else:
            print(f'invalid type ({gamma_type}) of gamma. type must be float or list')
            return False
        return True

    def fit(self):
        gflops = ModelUtil.get_gflops(self.__model)
        self.__model.save('model.h5', include_optimizer=False)
        self.__model.summary()
        print(f'\nGFLOPs : {gflops:.4f}')
        print(f'\ntrain on {len(self.__train_image_paths)} samples.')
        print(f'validate on {len(self.__validation_image_paths)} samples.')

        print('\ninvalid label check in train data...')
        self.__train_data_generator_for_check.flow().check_invalid_label()
        print('\ninvalid label check in validation data...')
        self.__validation_data_generator_for_check.flow().check_invalid_label()
        print('\ncalculate virtual anchor...')
        self.__train_data_generator.flow().calculate_virtual_anchor()
        print('\ncalculate BPR(Best Possible Recall)...')
        self.__train_data_generator.flow().calculate_best_possible_recall()
        if not self.__set_alpha_gamma():
            return
        print('\nstart test forward for checking forwarding time.')
        if ModelUtil.available_device() == 'gpu':
            ModelUtil.check_forwarding_time(self.__model, device='gpu')
        if tf.keras.backend.image_data_format() == 'channels_last':  # default max pool 2d layer is run on gpu only
            ModelUtil.check_forwarding_time(self.__model, device='cpu')

        print(f'\nalpha : {self.__alphas}')
        print(f'gamma : {self.__gammas}')
        print('\nstart training')
        if self.__curriculum_iterations > 0:
            self.__curriculum_train()
        self.__train()

    def compute_gradient(self, model, optimizer, loss_function, x, y_true, num_output_layers, alphas, gammas, label_smoothing):
        with tf.GradientTape() as tape:
            loss = 0.0
            y_pred = model(x, training=True)
            if num_output_layers == 1:
                loss = loss_function(y_true, y_pred, alphas[0], gammas[0], label_smoothing)
            else:
                for i in range(num_output_layers):
                    loss += loss_function(y_true[i], y_pred[i], alphas[i], gammas[i], label_smoothing)
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def __refresh_model_and_optimizer(self, model, optimizer_str):
        sleep(0.2)
        model.save('model.h5', include_optimizer=False)
        sleep(0.2)
        model = tf.keras.models.load_model('model.h5', compile=False)
        optimizer = self.__get_optimizer(optimizer_str)
        return model, optimizer

    def __curriculum_train(self):
        loss_functions = [confidence_loss, confidence_with_bbox_loss]
        compute_gradients = [tf.function(self.compute_gradient) for _ in range(len(loss_functions))]
        lr_scheduler = LRScheduler(iterations=self.__curriculum_iterations, lr=self.__lr)
        for i in range(len(loss_functions)):
            iteration_count = 0
            self.__model, optimizer = self.__refresh_model_and_optimizer(self.__model, 'sgd')
            while True:
                for batch_x, batch_y in self.__train_data_generator.flow():
                    iteration_count += 1
                    lr_scheduler.update(optimizer, iteration_count, self.__burn_in, 'onecycle')
                    loss = compute_gradients[i](self.__model, optimizer, loss_functions[i], batch_x, batch_y, self.num_output_layers, self.__alphas, self.__gammas, self.__label_smoothing)
                    print(f'\r[curriculum iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                    if iteration_count == self.__curriculum_iterations:
                        print()
                        break
                if iteration_count == self.__curriculum_iterations:
                    break

    def __train(self):
        iteration_count = 0
        compute_gradient_tf = tf.function(self.compute_gradient)
        self.__model, optimizer = self.__refresh_model_and_optimizer(self.__model, self.__optimizer)
        lr_scheduler = LRScheduler(iterations=self.__iterations, lr=self.__lr)
        while True:
            for batch_x, batch_y in self.__train_data_generator.flow():
                lr_scheduler.update(optimizer, iteration_count, self.__burn_in, self.__lr_policy)
                loss = compute_gradient_tf(self.__model, optimizer, yolo_loss, batch_x, batch_y, self.num_output_layers, self.__alphas, self.__gammas, self.__label_smoothing)
                iteration_count += 1
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                if self.__training_view and iteration_count > self.__burn_in:
                    self.__training_view_function()
                if self.__map_checkpoint:
                    if iteration_count >= int(self.__iterations * 0.7) and iteration_count % 20000 == 0:
                    # if iteration_count == self.__iterations:
                    # if iteration_count % 1000 == 0:
                    # if iteration_count >= (self.__iterations * 0.1) and iteration_count % 5000 == 0:
                        self.__save_model(iteration_count=iteration_count, use_map_checkpoint=self.__map_checkpoint)
                else:
                    if iteration_count % 10000 == 0:
                        self.__save_model(iteration_count=iteration_count, use_map_checkpoint=self.__map_checkpoint)
                if iteration_count % 1000 == 0:
                    self.__model.save('model_last.h5', include_optimizer=False)
                if iteration_count == self.__iterations:
                    print('\n\ntrain end successfully')
                    return

    def __save_model(self, iteration_count, use_map_checkpoint):
        print('\n')
        if use_map_checkpoint:
            self.calculate_map(dataset='validation', iteration_count=iteration_count, save_model=True)
        else:
            self.__model.save(f'{self.__checkpoints}/{self.__model_name}_{iteration_count}_iter.h5', include_optimizer=False)

    @staticmethod
    def predict(model, img, device, confidence_threshold=0.25, nms_iou_threshold=0.45, verbose=False):
        """
        Detect object in image using trained YOLO model.
        :param img: (width, height, channel) formatted image to be predicted.
        :param confidence_threshold: threshold confidence score to detect object.
        :param nms_iou_threshold: threshold to remove overlapped detection.
        :return: dictionary array sorted by x position.
        each dictionary has class index and bbox info: [x1, y1, x2, y2].
        """
        raw_width, raw_height = img.shape[1], img.shape[0]
        input_shape = model.input_shape[1:]
        input_width, input_height, _ = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        output_shape = model.output_shape
        num_output_layers = 1 if type(output_shape) == tuple else len(output_shape)
        if num_output_layers == 1:
            output_shape = [output_shape]

        img = ModelUtil.resize(img, (input_width, input_height))
        x = ModelUtil.preprocess(img)
        x = np.reshape(x, (1,) + input_shape)
        y = ModelUtil.graph_forward(model, x, device)
        y = np.array(y)
        if num_output_layers == 1:
            y = [y]

        bbox_count = 0
        y_pred = []
        image_data_format = tf.keras.backend.image_data_format()
        for layer_index in range(num_output_layers):
            rows = output_shape[layer_index][1]
            cols = output_shape[layer_index][2]
            cur_layer_output = y[layer_index][0]
            over_confidence_indexes = np.argwhere(cur_layer_output[:, :, 0] > confidence_threshold)
            for i, j in over_confidence_indexes:
                confidence = cur_layer_output[i][j][0]
                class_index = -1
                class_score = 0.0
                for cur_channel_index in range(5, output_shape[layer_index][3]):
                    cur_class_score = cur_layer_output[i][j][cur_channel_index]
                    if class_score < cur_class_score:
                        class_index = cur_channel_index
                        class_score = cur_class_score

                confidence *= class_score
                if confidence < confidence_threshold:
                    continue

                if image_data_format == 'channels_first':
                    cx_f = (j + cur_layer_output[1][i][j]) / float(cols)
                    cy_f = (i + cur_layer_output[2][i][j]) / float(rows)
                    w = cur_layer_output[3][i][j]
                    h = cur_layer_output[4][i][j]
                else:
                    cx_f = (j + cur_layer_output[i][j][1]) / float(cols)
                    cy_f = (i + cur_layer_output[i][j][2]) / float(rows)
                    w = cur_layer_output[i][j][3]
                    h = cur_layer_output[i][j][4]
                cx_f, cy_f, w, h = np.clip(np.array([cx_f, cy_f, w, h]), 0.0, 1.0)

                x_min_f = cx_f - (w * 0.5)
                y_min_f = cy_f - (h * 0.5)
                x_max_f = cx_f + (w * 0.5)
                y_max_f = cy_f + (h * 0.5)
                x_min_f, y_min_f, x_max_f, y_max_f = np.clip(np.array([x_min_f, y_min_f, x_max_f, y_max_f]), 0.0, 1.0)
                x_min = int(x_min_f * raw_width)
                y_min = int(y_min_f * raw_height)
                x_max = int(x_max_f * raw_width)
                y_max = int(y_max_f * raw_height)
                y_pred.append({
                    'confidence': confidence,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'bbox_norm': [x_min_f, y_min_f, x_max_f, y_max_f],
                    'class': class_index - 5,
                    'discard': False})
                bbox_count += 1
        y_pred = ModelUtil.nms(y_pred, nms_iou_threshold)
        if verbose:
            print(f'before nms box count : {bbox_count}')
            print(f'after  nms box count : {len(y_pred)}')
            print()
            for box_info in y_pred:
                class_index = box_info['class']
                confidence = box_info['confidence']
                bbox = box_info['bbox']
                bbox_norm = box_info['bbox_norm']
                print(f'class index : {class_index}')
                print(f'confidence : {confidence:.4f}')
                print(f'bbox : {np.array(bbox)}')
                print(f'bbox(normalized) : {np.array(bbox_norm)}')
                print()
            print()
        return y_pred

    def bounding_box(self, img, yolo_res, font_scale=0.4):
        """
        draw bounding bbox using result of YOLO.predict function.
        :param img: image to be predicted.
        :param yolo_res: result value of YOLO.predict() function.
        :param font_scale: scale of font.
        :return: image of bounding boxed.
        """
        padding = 5
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_height, img_width = img.shape[:2]
        for i, cur_res in enumerate(yolo_res):
            class_index = int(cur_res['class'])
            if len(self.__class_names) == 0:
                class_name = str(class_index)
            else:
                class_name = self.__class_names[class_index].replace('\n', '')
            label_background_color = colors[class_index]
            label_font_color = (0, 0, 0) if ModelUtil.is_background_color_bright(label_background_color) else (255, 255, 255)
            label_text = f'{class_name}({int(cur_res["confidence"] * 100.0)}%)'
            x1, y1, x2, y2 = cur_res['bbox_norm']
            x1 = int(x1 * img_width)
            y1 = int(y1 * img_height)
            x2 = int(x2 * img_width)
            y2 = int(y2 * img_height)
            l_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
            bw, bh = l_size[0] + (padding * 2), l_size[1] + (padding * 2) + baseline
            cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 1)
            cv2.rectangle(img, (x1 - 1, y1 - bh), (x1 - 1 + bw, y1), label_background_color, -1)
            cv2.putText(img, label_text, (x1 + padding - 1, y1 - baseline - padding), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
        return img

    def predict_video(self, video_path):
        """
        Equal to the evaluate function. video path is required.
        """
        cap = cv2.VideoCapture(video_path)
        while True:
            frame_exist, raw = cap.read()
            if not frame_exist:
                break
            x = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if self.__model.input.shape[-1] == 1 else raw.copy()
            res = Yolo.predict(self.__model, x, device='auto')
            # raw = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
            boxed_image = self.bounding_box(raw, res)
            cv2.imshow('video', boxed_image)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
        cap.release()
        cv2.destroyAllWindows()

    def predict_images(self, dataset='validation'):
        """
        Equal to the evaluate function. image paths are required.
        """
        input_width, input_height, input_channel = ModelUtil.get_width_height_channel_from_input_shape(self.__model.input_shape[1:])
        if dataset == 'train':
            image_paths = self.__train_image_paths
        elif dataset == 'validation':
            image_paths = self.__validation_image_paths
        else:
            print(f'invalid dataset : [{dataset}]')
            return
        for path in image_paths:
            print(f'image path : {path}')
            raw, raw_bgr, _ = ModelUtil.load_img(path, input_channel)
            res = Yolo.predict(self.__model, raw, device='cpu', verbose=True)
            raw_bgr = cv2.resize(raw_bgr, (input_width, input_height), interpolation=cv2.INTER_AREA)
            boxed_image = self.bounding_box(raw_bgr, res)
            cv2.imshow('res', boxed_image)
            key = cv2.waitKey(0)
            if key == 27:
                break

    def calculate_map(self, dataset='validation', iteration_count=0, save_model=False, device='auto'):
        if dataset == 'train':
            image_paths = self.__train_image_paths
        elif dataset == 'validation':
            image_paths = self.__validation_image_paths
        else:
            print(f'invalid dataset : [{dataset}]')
            return
        device = ModelUtil.available_device() if device == 'auto' else device
        mean_ap, f1_score, iou, tp, fp, fn, confidence = calc_mean_average_precision(self.__model, image_paths, device=device)
        if save_model:
            model_path = f'{self.__checkpoints}/'
            model_path += f'{self.__model_name}'
            if iteration_count > 0:
                model_path += f'_{iteration_count}_iter'
            model_path += f'_mAP_{mean_ap:.4f}'
            model_path += f'_f1_{f1_score:.4f}'
            model_path += f'_iou_{iou:.4f}'
            model_path += f'_tp_{tp}_fp_{fp}_fn_{fn}'
            model_path += f'_conf_{confidence:.4f}'
            model_path += f'.h5'
            self.__model.save(model_path, include_optimizer=False)
            print(f'model saved to [{model_path}]')

    def __training_view_function(self):
        """
        During training, the image is forwarded in real time, showing the results are shown.
        """
        cur_time = time()
        if cur_time - self.__live_view_previous_time > 0.5:
            self.__live_view_previous_time = cur_time
            if np.random.uniform() > 0.5:
                img_path = np.random.choice(self.__train_image_paths)
            else:
                img_path = np.random.choice(self.__validation_image_paths)
            img, raw_bgr, _ = ModelUtil.load_img(img_path, self.__input_channel)
            boxes = Yolo.predict(self.__model, img, device='cpu')
            boxed_image = self.bounding_box(raw_bgr, boxes)
            cv2.imshow('training view', boxed_image)
            cv2.waitKey(1)

