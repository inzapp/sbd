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
import random
from glob import glob
from time import time, sleep, perf_counter

import natsort
import numpy as np
import tensorflow as tf
from cv2 import cv2
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from box_colors import colors
from generator import YoloDataGenerator
from loss import confidence_loss, confidence_with_bbox_loss, yolo_loss
from mAP_calculator import calc_mean_average_precision
from model import Model


class Yolo:
    def __init__(self,
                 train_image_path=None,
                 input_shape=(256, 256, 1),
                 lr=0.001,
                 decay=0.0005,
                 momentum=0.9,
                 burn_in=1000,
                 batch_size=4,
                 iterations=100000,
                 curriculum_iterations=0,
                 validation_split=0.2,
                 validation_image_path='',
                 optimizer='sgd',
                 lr_policy='step',
                 test_only=False,
                 training_view=False,
                 map_checkpoint=False,
                 mixed_float16_training=False,
                 pretrained_model_path='',
                 class_names_file_path=''):
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
        self.__curriculum_iterations = curriculum_iterations
        self.__mixed_float16_training = mixed_float16_training
        self.__live_view_previous_time = time()
        self.__cycle_step = 0
        self.__cycle_length = 2500
        self.max_map, self.max_f1, self.max_map_iou_hm, self.max_f1_iou_hm = 0.0, 0.0, 0.0, 0.0

        if class_names_file_path == '':
            class_names_file_path = f'{train_image_path}/classes.txt'
        self.__class_names, self.__num_classes = self.__init_class_names(class_names_file_path)

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

        if validation_image_path != '':
            self.__train_image_paths, _ = self.__init_image_paths(train_image_path)
            self.__validation_image_paths, _ = self.__init_image_paths(validation_image_path)
        elif validation_split > 0.0:
            self.__train_image_paths, self.__validation_image_paths = self.__init_image_paths(train_image_path, validation_split=validation_split)

        if test_only:
            return

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

        self.__live_loss_plot = None
        if self.__mixed_float16_training:
            mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))
        os.makedirs('checkpoints', exist_ok=True)

    def __get_optimizer(self, optimizer_str):
        if optimizer_str == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=self.__lr, momentum=self.__momentum, nesterov=True)
        elif optimizer_str == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=self.__lr, beta_1=self.__momentum)
        else:
            print(f'\n\nunknown optimizer : {optimizer_str}')
            return None
        if self.__mixed_float16_training:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer=optimizer, loss_scale='dynamic')
        return optimizer

    def fit(self):
        self.__model.summary()
        self.__model.save('model.h5', include_optimizer=False)
        print(f'\ntrain on {len(self.__train_image_paths)} samples.')
        print(f'validate on {len(self.__validation_image_paths)} samples.')

        print('start training')
        self.__train_data_generator.flow().cluster_wh()
        self.__train_data_generator.flow().start()
        self.__train_data_generator.flow().print_not_trained_box_count()
        optimizer = self.__get_optimizer(self.__optimizer)
        self.__model.compile(optimizer=optimizer, loss=yolo_loss)
        self.__check_forwarding_time()

        if self.__burn_in > 0:
            self.__burn_in_train()
        if self.__curriculum_iterations > 0:
            self.__curriculum_train()
        self.__train()

    def __check_forwarding_time(self):
        input_shape = self.__model.input_shape[1:]
        mul = 1
        for val in input_shape:
            mul *= val

        forward_count = 32
        noise = np.random.uniform(0.0, 1.0, mul * forward_count)
        noise = np.asarray(noise).reshape((forward_count, 1) + input_shape).astype('float32')
        with tf.device('/cpu:0'):
            self.__model.predict_on_batch(x=noise[0])  # only first forward is slow, skip first forward in check forwarding time

        print('\nstart test forward for check forwarding time.')
        with tf.device('/cpu:0'):
            st = perf_counter()
            for i in range(forward_count):
                self.__model.predict_on_batch(x=noise[i])
            et = perf_counter()
        forwarding_time = ((et - st) / forward_count) * 1000.0
        print(f'model forwarding time : {forwarding_time:.2f} ms')

    def __burn_in_train(self):
        optimizer = self.__get_optimizer('sgd')
        self.__model.compile(optimizer=optimizer, loss=yolo_loss)
        iteration_count = 0
        while True:
            for batch_x, batch_y in self.__train_data_generator.flow():
                iteration_count += 1
                self.__update_burn_in_lr(iteration_count=iteration_count)
                logs = self.__model.train_on_batch(batch_x, batch_y, return_dict=True)
                print(f'\r[burn in iteration count : {iteration_count:6d}] loss => {logs["loss"]:.4f}', end='')
                if iteration_count == self.__burn_in:
                    print()
                    self.__model.save('model.h5', include_optimizer=False)
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
        sleep(0.5)
        self.__model = tf.keras.models.load_model('model.h5', compile=False)
        optimizer = self.__get_optimizer(self.__optimizer)
        self.__model.compile(optimizer=optimizer, loss=yolo_loss)
        iteration_count = 0
        while True:
            for batch_x, batch_y in self.__train_data_generator.flow():
                if self.__lr_policy == 'cosine':
                    self.__update_cosine_lr()
                logs = self.__model.train_on_batch(batch_x, batch_y, return_dict=True)
                iteration_count += 1
                print(f'\r[iteration count : {iteration_count:6d}] loss => {logs["loss"]:.4f}', end='')
                if self.__training_view:
                    self.__training_view_function()

                # if iteration_count > int(self.__iterations * 0.9) and iteration_count % 2000 == 0:
                if iteration_count % 1000 == 0:
                    self.__save_model(iteration_count=iteration_count, use_map_checkpoint=self.__map_checkpoint)
                elif iteration_count % 20000 == 0:
                    self.__save_model(iteration_count=iteration_count, use_map_checkpoint=False)
                if self.__lr_policy == 'step':
                    if iteration_count == int(self.__iterations * 0.8):
                        tf.keras.backend.set_value(self.__model.optimizer.lr, self.__model.optimizer.lr * 0.1)
                    elif iteration_count == int(self.__iterations * 0.9):
                        tf.keras.backend.set_value(self.__model.optimizer.lr, self.__model.optimizer.lr * 0.1)
                if iteration_count == self.__iterations:
                    print('\n\ntrain end successfully')
                    return

    def __update_burn_in_lr(self, iteration_count):
        lr = self.__lr * pow(float(iteration_count) / self.__burn_in, 4)
        tf.keras.backend.set_value(self.__model.optimizer.lr, lr)

    def __update_cosine_lr(self):
        if self.__cycle_step % self.__cycle_length == 0 and self.__cycle_step != 0:
            self.__cycle_step = 0
            self.__cycle_length *= 2
        max_lr = self.__lr
        min_lr = 0.0
        # min_lr = self.__lr * 0.01
        # lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(((1.0 / (0.5 * self.__cycle_length)) * np.pi * self.__cycle_step) + np.pi))  # up and down
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(((1.0 / self.__cycle_length) * np.pi * (self.__cycle_step % self.__cycle_length))))  # down and down
        tf.keras.backend.set_value(self.__model.optimizer.lr, lr)
        self.__cycle_step += 1

    @staticmethod
    def __harmonic_mean(a, b):
        return (2.0 * a * b) / (a + b + 1e-5)

    def __is_better_than_before(self, mean_ap, f1_score, tp_iou):
        better_than_before = False
        if mean_ap > self.max_map:
            self.max_map = mean_ap
            better_than_before = True

        if f1_score > self.max_f1:
            self.max_f1 = f1_score
            better_than_before = True

        map_iou_hm = self.__harmonic_mean(mean_ap, tp_iou)
        if map_iou_hm > self.max_map_iou_hm:
            self.max_map_iou_hm = map_iou_hm
            better_than_before = True

        f1_iou_hm = self.__harmonic_mean(f1_score, tp_iou)
        if f1_iou_hm > self.max_f1_iou_hm:
            self.max_f1_iou_hm = f1_iou_hm
            better_than_before = True
        return better_than_before

    def __save_model(self, iteration_count, use_map_checkpoint=True):
        print('\n')
        if use_map_checkpoint:
            self.__model.save('model.h5', include_optimizer=False)
            mean_ap, f1_score, tp_iou, tp, fp, fn = calc_mean_average_precision(self.__model, self.__validation_image_paths)
            if self.__is_better_than_before(mean_ap, f1_score, tp_iou):
                self.__model.save(f'checkpoints/model_{iteration_count}_iter_mAP_{mean_ap:.4f}_f1_{f1_score:.4f}_tp_iou_{tp_iou:.4f}_tp_{tp}_fp_{fp}_fn_{fn}.h5', include_optimizer=False)
                self.__model.save('model_last.h5', include_optimizer=False)
        else:
            self.__model.save(f'checkpoints/model_{iteration_count}_iter.h5')

    @staticmethod
    def __init_image_paths(image_path, validation_split=0.0):
        if image_path.endswith('.txt'):
            with open(image_path, 'rt') as f:
                all_image_paths = f.readlines()
            for i in range(len(all_image_paths)):
                all_image_paths[i] = all_image_paths[i].replace('\n', '')
        else:
            all_image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        random.shuffle(all_image_paths)
        num_train_images = int(len(all_image_paths) * (1.0 - validation_split))
        image_paths = all_image_paths[:num_train_images]
        validation_image_paths = all_image_paths[num_train_images:]
        return image_paths, validation_image_paths

    def __nms(self, y_pred, nms_iou_threshold):
        y_pred = sorted(y_pred, key=lambda x: x['confidence'], reverse=True)
        for i in range(len(y_pred) - 1):
            if y_pred[i]['discard']:
                continue
            for j in range(i + 1, len(y_pred)):
                if y_pred[j]['discard'] or y_pred[i]['class'] != y_pred[j]['class']:
                    continue
                if self.__iou(y_pred[i]['bbox'], y_pred[j]['bbox']) > nms_iou_threshold:
                    y_pred[j]['discard'] = True

        y_pred_copy = np.asarray(y_pred.copy())
        y_pred = []
        for i in range(len(y_pred_copy)):
            if not y_pred_copy[i]['discard']:
                y_pred.append(y_pred_copy[i])
        return y_pred

    def predict(self, img, confidence_threshold=0.25, nms_iou_threshold=0.45):
        """
        Detect object in image using trained YOLO model.
        :param img: (width, height, channel) formatted image to be predicted.
        :param confidence_threshold: threshold confidence score to detect object.
        :param nms_iou_threshold: threshold to remove overlapped detection.
        :return: dictionary array sorted by x position.
        each dictionary has class index and bbox info: [x1, y1, x2, y2].
        """
        raw_width, raw_height = img.shape[1], img.shape[0]
        input_shape = self.__model.input_shape[1:]
        output_shape = self.__model.output_shape

        if img.shape[1] > input_shape[1] or img.shape[0] > input_shape[0]:
            img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)

        x = np.asarray(img).reshape((1,) + input_shape).astype('float32') / 255.0
        y = self.__model.predict_on_batch(x=x)

        y_pred = []
        for layer_index in range(len(output_shape)):
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

                    confidence = confidence * class_score
                    if confidence < confidence_threshold:
                        continue

                    cx_f = (j + y[layer_index][0][i][j][1]) / float(cols)
                    cy_f = (i + y[layer_index][0][i][j][2]) / float(rows)
                    w = y[layer_index][0][i][j][3]
                    h = y[layer_index][0][i][j][4]

                    x_min_f = cx_f - w / 2.0
                    y_min_f = cy_f - h / 2.0
                    x_max_f = cx_f + w / 2.0
                    y_max_f = cy_f + h / 2.0
                    x_min = int(x_min_f * raw_width)
                    y_min = int(y_min_f * raw_height)
                    x_max = int(x_max_f * raw_width)
                    y_max = int(y_max_f * raw_height)
                    y_pred.append({
                        'confidence': confidence,
                        'bbox': [x_min, y_min, x_max, y_max],
                        'class': class_index - 5,
                        'discard': False})

        y_pred = self.__nms(y_pred, nms_iou_threshold)
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
        for i, cur_res in enumerate(yolo_res):
            class_index = int(cur_res['class'])
            if len(self.__class_names) == 0:
                class_name = str(class_index)
            else:
                class_name = self.__class_names[class_index].replace('\n', '')
            label_background_color = colors[class_index]
            label_font_color = (0, 0, 0) if self.__is_background_color_bright(label_background_color) else (255, 255, 255)
            label_text = f'{class_name}({int(cur_res["confidence"] * 100.0)}%)'
            x1, y1, x2, y2 = cur_res['bbox']
            l_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
            bw, bh = l_size[0] + (padding * 2), l_size[1] + (padding * 2) + baseline
            cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 1)
            # cv2.rectangle(img, (x1 - 1, y1 - bh), (x1 - 1 + bw, y1), label_background_color, -1)
            # cv2.putText(img, label_text, (x1 + padding - 1, y1 - baseline - padding), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
        return img

    def predict_video(self, video_path):
        """
        Equal to the evaluate function. video path is required.
        """
        with tf.device('/cpu:0'):
            cap = cv2.VideoCapture(video_path)
            while True:
                frame_exist, raw = cap.read()
                if not frame_exist:
                    break
                x = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if self.__model.input.shape[-1] == 1 else raw.copy()
                res = self.predict(x)
                boxed_image = self.bounding_box(raw, res)
                cv2.imshow('video', boxed_image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == 27:
                    exit(0)
            cap.release()
            cv2.destroyAllWindows()

    def predict_images(self, image_paths):
        """
        Equal to the evaluate function. image paths are required.
        """
        if type(image_paths) is str:
            image_paths = glob(image_paths)
        image_paths = natsort.natsorted(image_paths)
        with tf.device('/cpu:0'):
            for path in image_paths:
                raw = cv2.imread(path, cv2.IMREAD_COLOR)
                x = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if self.__model.input.shape[-1] == 1 else raw.copy()
                res = self.predict(x)
                boxed_image = self.bounding_box(raw, res)
                cv2.imshow('res', boxed_image)
                key = cv2.waitKey(0)
                if key == 27:
                    break

    def predict_train_images(self):
        self.predict_images(self.__train_image_paths)

    def predict_validation_images(self):
        self.predict_images(self.__validation_image_paths)

    def map_train_images(self):
        calc_mean_average_precision(self.__model, self.__train_image_paths)

    def map_validation_images(self):
        calc_mean_average_precision(self.__model, self.__validation_image_paths)

    def __training_view_function(self):
        """
        During training, the image is forwarded in real time, showing the results are shown.
        """
        cur_time = time()
        if cur_time - self.__live_view_previous_time > 0.5:
            self.__live_view_previous_time = cur_time
            if np.random.uniform() > 0.5:
                img_path = random.choice(self.__train_image_paths)
            else:
                img_path = random.choice(self.__validation_image_paths)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if self.__model.input.shape[-1] == 1 else cv2.IMREAD_COLOR)
            res = self.predict(img)
            boxed_image = self.bounding_box(img, res)
            cv2.imshow('training view', boxed_image)
            cv2.waitKey(1)

    @staticmethod
    def __init_class_names(class_names_file_path):
        """
        Init YOLO label from classes.txt file.
        If the class file is not found, it is replaced by class index and displayed.
        """
        if os.path.exists(class_names_file_path) and os.path.isfile(class_names_file_path):
            with open(class_names_file_path, 'rt') as classes_file:
                class_names = [s.replace('\n', '') for s in classes_file.readlines()]
                num_classes = len(class_names)
            return class_names, num_classes
        else:
            print(f'class names file dose not exist : {class_names_file_path}')
            print('class file does not exist. the class name will be replaced by the class index and displayed.')
            return [], 0

    @staticmethod
    def __iou(a, b):
        """
        Intersection of union function.
        :param a: [x_min, y_min, x_max, y_max] format box a
        :param b: [x_min, y_min, x_max, y_max] format box b
        """
        a_x_min, a_y_min, a_x_max, a_y_max = a
        b_x_min, b_y_min, b_x_max, b_y_max = b
        intersection_width = min(a_x_max, b_x_max) - max(a_x_min, b_x_min)
        intersection_height = min(a_y_max, b_y_max) - max(a_y_min, b_y_min)
        if intersection_width <= 0 or intersection_height <= 0:
            return 0.0
        intersection_area = intersection_width * intersection_height
        a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
        b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
        union_area = a_area + b_area - intersection_area
        return intersection_area / (float(union_area) + 1e-5)

    @staticmethod
    def __is_background_color_bright(bgr):
        """
        Determine whether the color is bright or not.
        :param bgr: bgr scalar tuple.
        :return: true if parameter color is bright and false if not.
        """
        tmp = np.zeros((1, 1), dtype=np.uint8)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tmp, (0, 0), (1, 1), bgr, -1)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        return tmp[0][0] > 127
