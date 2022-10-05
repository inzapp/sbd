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
from glob import glob
from time import time, sleep, perf_counter

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from util import ModelUtil
from box_colors import colors
from generator import YoloDataGenerator
from generator import GeneratorFlow
from loss import confidence_loss, confidence_with_bbox_loss, yolo_loss
from mAP_calculator import calc_mean_average_precision
from model import Model
from keras_flops import get_flops


class Yolo:
    g_use_layers = []
    def __init__(self,
                 train_image_path=None,
                 input_shape=(256, 256, 1),
                 lr=0.001,
                 decay=0.0005,
                 momentum=0.9,
                 burn_in=1000,
                 batch_size=4,
                 iterations=100000,
                 ignore_threshold=0.9,
                 curriculum_iterations=0,
                 validation_split=0.2,
                 validation_image_path='',
                 optimizer='sgd',
                 lr_policy='step',
                 use_layers=[],
                 training_view=False,
                 map_checkpoint=False,
                 mixed_float16_training=False,
                 pretrained_model_path='',
                 class_names_file_path='',
                 checkpoints='checkpoints'):
        self.__lr = lr
        self.__decay = decay
        self.__momentum = momentum
        self.__burn_in = burn_in
        self.__batch_size = batch_size
        self.__iterations = iterations
        self.__optimizer = optimizer
        self.__lr_policy = lr_policy
        self.__training_view = training_view
        self.__map_checkpoint = map_checkpoint
        self.__ignore_threshold = ignore_threshold
        self.__curriculum_iterations = curriculum_iterations
        self.__mixed_float16_training = mixed_float16_training
        self.__live_view_previous_time = time()
        self.__checkpoints = checkpoints
        self.__cycle_step = 0
        self.__cycle_length = 2500
        self.max_map, self.max_f1, self.max_map_iou_hm, self.max_f1_iou_hm = 0.0, 0.0, 0.0, 0.0
        Yolo.g_use_layers = use_layers

        self.__input_width, self.__input_height, self.__input_channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        ModelUtil.set_channel_order(input_shape)

        if class_names_file_path == '':
            class_names_file_path = f'{train_image_path}/classes.txt'
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
            batch_size=batch_size)
        self.__validation_data_generator = YoloDataGenerator(
            image_paths=self.__validation_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=batch_size)
        self.__train_data_generator_for_check = YoloDataGenerator(
            image_paths=self.__train_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=ModelUtil.get_zero_mod_batch_size(len(self.__train_image_paths)))
        self.__validation_data_generator_for_check = YoloDataGenerator(
            image_paths=self.__validation_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=ModelUtil.get_zero_mod_batch_size(len(self.__validation_image_paths)))

        self.__live_loss_plot = None
        if self.__mixed_float16_training:
            mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))
        os.makedirs(f'{self.__checkpoints}', exist_ok=True)

    def __get_optimizer(self, optimizer_str):
        if optimizer_str == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=1.0, momentum=self.__momentum, nesterov=True)
        elif optimizer_str == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=self.__lr, beta_1=self.__momentum)
        else:
            print(f'\n\nunknown optimizer : {optimizer_str}')
            return None
        if self.__mixed_float16_training:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer=optimizer, loss_scale='dynamic')
        return optimizer

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
        print('\nnot assigned bbox counting in train tensor...')
        self.__train_data_generator_for_check.flow().print_not_trained_box_count()
        print('\nstart test forward for checking forwarding time.')
        ModelUtil.check_forwarding_time(self.__model, device='gpu')
        if tf.keras.backend.image_data_format() == 'channels_last':  # default max pool 2d layer is run on gpu only
            ModelUtil.check_forwarding_time(self.__model, device='cpu')

        print('\nstart training')
        if self.__burn_in > 0 and self.__optimizer == 'sgd':
            self.__burn_in_train()
        if self.__curriculum_iterations > 0:
            self.__curriculum_train()
        self.__train()

    def __burn_in_train(self):
        @tf.function
        def compute_gradient(model, optimizer, x, y_true, lr, num_output_layers, ignore_threshold):
            with tf.GradientTape() as tape:
                loss = 0.0
                y_pred = model(x, training=True)
                if num_output_layers == 1:
                    loss = yolo_loss(y_true, y_pred, ignore_threshold=ignore_threshold)
                else:
                    for i in range(num_output_layers):
                        loss += yolo_loss(y_true[i], y_pred[i], ignore_threshold=ignore_threshold)
                lr = tf.cast(lr, dtype=loss.dtype)
                gradients = tape.gradient(loss * lr, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        optimizer = self.__get_optimizer('sgd')
        iteration_count = 0
        lr = 0.0
        while True:
            for batch_x, batch_y in self.__train_data_generator.flow():
                iteration_count += 1
                lr = self.__update_burn_in_lr(iteration_count=iteration_count)
                loss = compute_gradient(self.__model, optimizer, batch_x, batch_y, tf.constant(lr), self.num_output_layers, self.__ignore_threshold)
                print(f'\r[burn in iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                if iteration_count == self.__burn_in:
                    print()
                    return

    def __curriculum_train(self):
        sleep(0.5)
        self.__model = tf.keras.models.load_model('model.h5', compile=False)
        tmp_model_name = f'{time()}.h5'
        for loss in [confidence_loss, confidence_with_bbox_loss]:
            optimizer = self.__get_optimizer(self.__optimizer)
            self.__model.compile(optimizer=optimizer, loss=loss)
            iteration_count = 0
            while True:
                for batch_x, batch_y in self.__train_data_generator.flow():
                    iteration_count += 1
                    logs = self.__model.train_on_batch(batch_x, batch_y, return_dict=True)
                    print(f'\r[curriculum iteration count : {iteration_count:6d}] loss => {logs["loss"]:.4f}', end='')
                    if iteration_count == self.__curriculum_iterations:
                        print()
                        self.__model.save(tmp_model_name, include_optimizer=False)
                        sleep(0.5)
                        self.__model = tf.keras.models.load_model(tmp_model_name, compile=False)
                        sleep(0.5)
                        os.remove(tmp_model_name)
                        return

    def __train(self):
        @tf.function
        def compute_gradient(model, optimizer, x, y_true, lr, num_output_layers, is_sgd):
            with tf.GradientTape() as tape:
                loss = 0.0
                y_pred = model(x, training=True)
                if num_output_layers == 1:
                    loss = yolo_loss(y_true, y_pred)
                else:
                    for i in range(num_output_layers):
                        loss += yolo_loss(y_true[i], y_pred[i])
                if is_sgd:
                    loss *= tf.cast(lr, dtype=loss.dtype)
                gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        lr = self.__lr
        cosine_save = True
        iteration_count = 0
        optimizer = self.__get_optimizer(self.__optimizer)
        b_sgd = optimizer.__str__().lower().find('sgd') > -1
        while True:
            for batch_x, batch_y in self.__train_data_generator.flow():
                if self.__lr_policy == 'cosine':
                    lr = self.__update_cosine_lr()
                    if lr == self.__lr:
                        cosine_save = True
                loss = compute_gradient(self.__model, optimizer, batch_x, batch_y, tf.constant(lr), self.num_output_layers, b_sgd)
                iteration_count += 1
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                if self.__training_view:
                    self.__training_view_function()

                if self.__lr_policy == 'cosine':
                    if cosine_save and lr < 1e-9:
                        self.__save_model(iteration_count=iteration_count, use_map_checkpoint=self.__map_checkpoint)
                        cosine_save = False
                elif self.__lr_policy == 'step':
                    if iteration_count == int(self.__iterations * 0.8):
                        lr *= 0.1
                    elif iteration_count == int(self.__iterations * 0.9):
                        lr *= 0.1
                    if iteration_count > int(self.__iterations * 0.5) and iteration_count % 10000 == 0:
                    # if iteration_count % 1000 == 0:
                    # if iteration_count == self.__iterations:
                        self.__save_model(iteration_count=iteration_count, use_map_checkpoint=self.__map_checkpoint)
                elif self.__lr_policy == 'constant':
                    if iteration_count % 10000 == 0:
                        self.__save_model(iteration_count=iteration_count, use_map_checkpoint=True)
                else:
                    print(f'unknwon lr policy : {self.__lr_policy}')
                    exit(0)
                if iteration_count == self.__iterations:
                    print('\n\ntrain end successfully')
                    return

    def __update_burn_in_lr(self, iteration_count):
        return self.__lr * pow(float(iteration_count) / self.__burn_in, 4)

    def __update_cosine_lr(self):
        if self.__cycle_step % self.__cycle_length == 0 and self.__cycle_step != 0:
            self.__cycle_step = 0
            self.__cycle_length *= 2
        max_lr = self.__lr
        min_lr = 0.0
        # min_lr = self.__lr * 0.01
        # lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(((1.0 / (0.5 * self.__cycle_length)) * np.pi * self.__cycle_step) + np.pi))  # up and down
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(((1.0 / self.__cycle_length) * np.pi * (self.__cycle_step % self.__cycle_length))))  # down and down
        self.__cycle_step += 1
        return lr

    def __save_model(self, iteration_count, use_map_checkpoint=True):
        print('\n')
        ul = str(Yolo.g_use_layers) if len(Yolo.g_use_layers) > 0 else 'all'
        if use_map_checkpoint:
            self.__model.save('model.h5', include_optimizer=False)
            mean_ap, f1_score, tp_iou, tp, fp, fn = calc_mean_average_precision(self.__model, self.__validation_image_paths)
            # if self.__is_better_than_before(mean_ap, f1_score, tp_iou):
            self.__model.save(f'{self.__checkpoints}/model_{iteration_count}_iter_mAP_{mean_ap:.4f}_f1_{f1_score:.4f}_tp_iou_{tp_iou:.4f}_tp_{tp}_fp_{fp}_fn_{fn}_ul_{ul}.h5', include_optimizer=False)
            self.__model.save(f'model_last_ul_{ul}.h5', include_optimizer=False)
        else:
            self.__model.save(f'{self.__checkpoints}/model_{iteration_count}_iter_ul_{ul}.h5', include_optimizer=False)

    @staticmethod
    def predict(model, img, device, confidence_threshold=0.25, nms_iou_threshold=0.45):
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
            if len(Yolo.g_use_layers) > 0 and layer_index not in Yolo.g_use_layers:
                continue
            rows = output_shape[layer_index][1]
            cols = output_shape[layer_index][2]
            for i in range(rows):
                for j in range(cols):
                    confidence = y[layer_index][0][i][j][0]
                    if confidence < confidence_threshold:
                        continue

                    class_index = -1
                    class_score = 0.0
                    for cur_channel_index in range(5, output_shape[layer_index][3]):
                        cur_class_score = y[layer_index][0][i][j][cur_channel_index]
                        if class_score < cur_class_score:
                            class_index = cur_channel_index
                            class_score = cur_class_score

                    confidence *= class_score
                    if confidence < confidence_threshold:
                        continue

                    if image_data_format == 'channels_first':
                        cx_f = (j + y[layer_index][0][1][i][j]) / float(cols)
                        cy_f = (i + y[layer_index][0][2][i][j]) / float(rows)
                        w = y[layer_index][0][3][i][j]
                        h = y[layer_index][0][4][i][j]
                    else:
                        cx_f = (j + y[layer_index][0][i][j][1]) / float(cols)
                        cy_f = (i + y[layer_index][0][i][j][2]) / float(rows)
                        w = y[layer_index][0][i][j][3]
                        h = y[layer_index][0][i][j][4]
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
        # print(f' detected box count, nms box count : [{bbox_count}, {len(y_pred)}]')
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
            res = Yolo.predict(self.__model, x, device='gpu')
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
            raw, raw_bgr, _ = ModelUtil.load_img(path, input_channel)
            res = Yolo.predict(self.__model, raw, device='cpu')
            boxed_image = self.bounding_box(raw_bgr, res)
            cv2.imshow('res', boxed_image)
            key = cv2.waitKey(0)
            if key == 27:
                break

    def calculate_map(self, dataset='validation'):
        if dataset == 'train':
            calc_mean_average_precision(self.__model, self.__train_image_paths)
        elif dataset == 'validation':
            calc_mean_average_precision(self.__model, self.__validation_image_paths)
        else:
            print(f'invalid dataset : [{dataset}]')
            return

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

