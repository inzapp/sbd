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

import cv2
import numpy as np
import tensorflow as tf

from model import Model
from util import ModelUtil
from box_colors import colors
from lr_scheduler import LRScheduler
from generator import YoloDataGenerator
from mAP_calculator import calc_mean_average_precision
from loss import confidence_loss, confidence_with_bbox_loss, yolo_loss


class Yolo:
    def __init__(self, config):
        pretrained_model_path = config['pretrained_model_path']
        input_shape = config['input_shape']
        train_image_path = config['train_image_path']
        validation_image_path = config['validation_image_path']
        multi_classification_at_same_box = config['multi_classification_at_same_box']
        ignore_nearby_cell = config['ignore_nearby_cell']
        nearby_cell_ignore_threshold = config['nearby_cell_ignore_threshold']
        batch_size = config['batch_size']
        self.__class_names_file_path = config['class_names_file_path']
        self.__lr = config['lr']
        self.__alpha_arg = config['alpha'] 
        self.__alphas = None
        self.__gamma_arg = config['gamma']
        self.__gammas = None
        self.__l2 = config['l2']
        self.__momentum = config['momentum']
        self.__label_smoothing = config['label_smoothing']
        self.__warm_up = config['warm_up']
        self.__decay_step = config['decay_step']
        self.__iterations = config['iterations']
        self.__optimizer = config['optimizer']
        self.__lr_policy = config['lr_policy']
        self.__model_name = config['model_name']
        self.__model_type = config['model_type']
        self.__training_view = config['training_view']
        self.__map_checkpoint = config['map_checkpoint']
        self.__curriculum_iterations = config['curriculum_iterations']
        self.__checkpoint_interval = config['checkpoint_interval']
        self.__live_view_previous_time = time()
        self.__checkpoint_path = config['checkpoint_path']
        self.__cycle_step = 0
        self.__cycle_length = 2500
        self.__presaved_iteration_count = 0
        self.max_map, self.max_f1, self.max_map_iou_hm, self.max_f1_iou_hm = 0.0, 0.0, 0.0, 0.0

        self.__input_width, self.__input_height, self.__input_channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
        self.__class_names, self.__num_classes = ModelUtil.init_class_names(self.__class_names_file_path)

        pretrained_model_load_success = False
        if pretrained_model_path != '':
            if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
                self.__model = tf.keras.models.load_model(pretrained_model_path, compile=False)
                self.__presaved_iteration_count = self.__parse_presaved_iteration_count(pretrained_model_path)
                pretrained_model_load_success = True
                print(f'success loading pretrained model : [{pretrained_model_path}]')
            else:
                ModelUtil.print_error_exit(f'pretrained model not found. model path : {pretrained_model_path}')

        if not pretrained_model_load_success:
            if self.__num_classes == 0:
                ModelUtil.print_error_exit(f'classes file not found. file path : {self.__class_names_file_path}')
            if self.__optimizer == 'adam':
                self.__l2 = 0.0
            self.__model = Model(input_shape=input_shape, output_channel=self.__num_classes + 5, l2=self.__l2).build(self.__model_type)

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
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_nearby_cell=ignore_nearby_cell,
            nearby_cell_ignore_threshold=nearby_cell_ignore_threshold)
        self.__validation_data_generator = YoloDataGenerator(
            image_paths=self.__validation_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=batch_size,
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_nearby_cell=ignore_nearby_cell,
            nearby_cell_ignore_threshold=nearby_cell_ignore_threshold)
        self.__train_data_generator_for_check = YoloDataGenerator(
            image_paths=self.__train_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=ModelUtil.get_zero_mod_batch_size(len(self.__train_image_paths)),
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_nearby_cell=ignore_nearby_cell,
            nearby_cell_ignore_threshold=nearby_cell_ignore_threshold)
        self.__validation_data_generator_for_check = YoloDataGenerator(
            image_paths=self.__validation_image_paths,
            input_shape=input_shape,
            output_shape=self.__model.output_shape,
            batch_size=ModelUtil.get_zero_mod_batch_size(len(self.__validation_image_paths)),
            multi_classification_at_same_box=multi_classification_at_same_box,
            ignore_nearby_cell=ignore_nearby_cell,
            nearby_cell_ignore_threshold=nearby_cell_ignore_threshold)

        self.__live_loss_plot = None
        os.makedirs(f'{self.__checkpoint_path}', exist_ok=True)
        np.set_printoptions(precision=6)

    def __parse_presaved_iteration_count(self, pretrained_model_path):
        iteration_count = 0
        if pretrained_model_path.find('iter') > -1:
            sp = pretrained_model_path.split('_') 
            for i in range(len(sp)):
                if sp[i] == 'iter' and i > 0:
                    try:
                        iteration_count = int(sp[i-1])
                    except ValueError:
                        iteration_count = 0
        return iteration_count

    def __get_optimizer(self, optimizer_str):
        lr = self.__lr if self.__lr_policy == 'constant' else 0.0
        if optimizer_str == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=self.__momentum, nesterov=True)
        elif optimizer_str == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=self.__momentum)
        elif optimizer_str == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
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

        print('\nlabel exist check in train data...')
        self.__train_data_generator_for_check.flow().check_labels_exist()
        print('\nlabel exist check in validation data...')
        self.__validation_data_generator_for_check.flow().check_labels_exist()
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
        ModelUtil.check_forwarding_time(self.__model, device='cpu')

        print(f'\nalpha : {self.__alphas}')
        print(f'gamma : {self.__gammas}')
        print('\nstart training')
        if self.__curriculum_iterations > 0:
            self.__curriculum_train()
        self.__train()

    def compute_gradient(self, model, optimizer, loss_function, x, y_true, mask, num_output_layers, alphas, gammas, label_smoothing):
        with tf.GradientTape() as tape:
            loss = 0.0
            y_pred = model(x, training=True)
            if num_output_layers == 1:
                confidence_loss, bbox_loss, classification_loss = loss_function(y_true, y_pred, mask, alphas[0], gammas[0], label_smoothing)
            else:
                confidence_loss, bbox_loss, classification_loss = 0.0, 0.0, 0.0
                for i in range(num_output_layers):
                    _confidence_loss, _bbox_loss, _classification_loss = loss_function(y_true[i], y_pred[i], mask[i], alphas[i], gammas[i], label_smoothing)
                    confidence_loss += _confidence_loss
                    bbox_loss += _bbox_loss
                    classification_loss += _classification_loss
            loss = confidence_loss + bbox_loss + classification_loss
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return confidence_loss, bbox_loss, classification_loss

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
                    lr_scheduler.update(optimizer, iteration_count, 'onecycle')
                    loss = compute_gradients[i](self.__model, optimizer, loss_functions[i], batch_x, batch_y, self.num_output_layers, self.__alphas, self.__gammas, self.__label_smoothing)
                    print(f'\r[curriculum iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                    if iteration_count == self.__curriculum_iterations:
                        print()
                        break
                if iteration_count == self.__curriculum_iterations:
                    break

    def build_loss_str(self, iteration_count, loss_vars):
        confidence_loss, bbox_loss, classification_loss = loss_vars
        loss_str = f'\r[iteration_count : {iteration_count:6d}]'
        loss_str += f' confidence_loss : {confidence_loss:>8.4f}'
        loss_str += f', bbox_loss: {bbox_loss:>8.4f}'
        loss_str += f', classification_loss : {classification_loss:>8.4f}'
        return loss_str

    def __train(self):
        iteration_count = 0
        compute_gradient_tf = tf.function(self.compute_gradient)
        self.__model, optimizer = self.__refresh_model_and_optimizer(self.__model, self.__optimizer)
        lr_scheduler = LRScheduler(iterations=self.__iterations, lr=self.__lr, warm_up=self.__warm_up, policy=self.__lr_policy, decay_step=self.__decay_step)
        while True:
            for batch_x, batch_y, mask in self.__train_data_generator.flow():
                lr_scheduler.update(optimizer, iteration_count)
                loss_vars = compute_gradient_tf(self.__model, optimizer, yolo_loss, batch_x, batch_y, mask, self.num_output_layers, self.__alphas, self.__gammas, self.__label_smoothing)
                iteration_count += 1
                print(self.build_loss_str(iteration_count, loss_vars), end='')
                warm_up_end = iteration_count >= int(self.__iterations * self.__warm_up)
                # warm_up_end = iteration_count >= int(self.__iterations * 0.7)
                if warm_up_end and self.__training_view:
                    self.__training_view_function()
                if self.__map_checkpoint:
                    if warm_up_end and iteration_count % self.__checkpoint_interval == 0 :
                        self.__save_model(iteration_count=iteration_count, use_map_checkpoint=self.__map_checkpoint)
                else:
                    if iteration_count % self.__checkpoint_interval == 0:
                        self.__save_model(iteration_count=iteration_count, use_map_checkpoint=self.__map_checkpoint)
                if iteration_count % 2000 == 0:
                    self.__model.save('model_last.h5', include_optimizer=False)
                if iteration_count == self.__iterations:
                    print('\n\ntrain end successfully')
                    return

    def __save_model(self, iteration_count, use_map_checkpoint):
        print('\n')
        if use_map_checkpoint:
            self.calculate_map(dataset='validation', iteration_count=iteration_count, save_model=True)
        else:
            self.__model.save(f'{self.__checkpoint_path}/{self.__model_name}_{iteration_count}_iter.h5', include_optimizer=False)

    @staticmethod
    def predict(model, img, device, confidence_threshold=0.2, nms_iou_threshold=0.45, verbose=False):
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

        output_tensor = y[0][0]
        output_tensor_rows = output_shape[0][1]
        output_tensor_cols = output_shape[0][2]

        cur_layer_output = output_tensor
        confidence_channel = output_tensor[:, :, 0]
        max_class_score_channel = np.max(output_tensor[:, :, 5:], axis=-1)
        max_class_index_channel = np.argmax(output_tensor[:, :, 5:], axis=-1)
        confidence_channel *= max_class_score_channel
        over_confidence_indexes = np.argwhere(confidence_channel > confidence_threshold)

        cx_channel = output_tensor[:, :, 1]
        cy_channel = output_tensor[:, :, 2]
        w_channel = output_tensor[:, :, 3]
        h_channel = output_tensor[:, :, 4]

        x_range = np.arange(output_tensor_cols, dtype=np.float32)
        x_offset = np.broadcast_to(x_range, shape=cx_channel.shape)

        y_range = np.arange(output_tensor_rows, dtype=np.float32)
        y_range = np.reshape(y_range, newshape=(output_tensor_rows, 1))
        y_offset = np.broadcast_to(y_range, shape=cy_channel.shape)

        cx_channel = (x_offset + cx_channel) / output_tensor_cols
        cy_channel = (y_offset + cy_channel) / output_tensor_rows

        xmin = cx_channel - (w_channel * 0.5)
        ymin = cy_channel - (h_channel * 0.5)
        xmax = cx_channel + (w_channel * 0.5)
        ymax = cy_channel + (h_channel * 0.5)

        boxes_before_nms = []
        for i, j in over_confidence_indexes:
            confidence = float(confidence_channel[i][j])
            class_index = int(max_class_index_channel[i][j])
            xmin_f = float(xmin[i][j])
            ymin_f = float(ymin[i][j])
            xmax_f = float(xmax[i][j])
            ymax_f = float(ymax[i][j])
            boxes_before_nms.append({
                'confidence': confidence,
                'bbox_norm': [xmin_f, ymin_f, xmax_f, ymax_f],
                'class': class_index,
                'discard': False})

        boxes = ModelUtil.nms(boxes_before_nms, nms_iou_threshold)
        if verbose:
            print(f'before nms box count : {len(boxes_before_nms)}')
            print(f'after  nms box count : {len(boxes)}')
            print()
            for box_info in boxes:
                class_index = box_info['class']
                confidence = box_info['confidence']
                bbox_norm = box_info['bbox_norm']
                print(f'class index : {class_index}')
                print(f'confidence : {confidence:.4f}')
                print(f'bbox : {np.array(bbox)}')
                print(f'bbox(normalized) : {np.array(bbox_norm)}')
                print()
            print()
        return boxes

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

    def predict_video(self, video_path, confidence_threshold=0.2, device='cpu'):
        """
        Equal to the evaluate function. video path is required.
        """
        cap = cv2.VideoCapture(video_path)
        while True:
            frame_exist, raw = cap.read()
            if not frame_exist:
                break
            x = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if self.__model.input.shape[-1] == 1 else raw.copy()
            res = Yolo.predict(self.__model, x, device=device, confidence_threshold=confidence_threshold)
            # raw = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
            boxed_image = self.bounding_box(raw, res)
            cv2.imshow('video', boxed_image)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
        cap.release()
        cv2.destroyAllWindows()

    def predict_images(self, dataset='validation', confidence_threshold=0.2, device='cpu'):
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
            res = Yolo.predict(self.__model, raw, device=device, verbose=True, confidence_threshold=confidence_threshold)
            raw_bgr = cv2.resize(raw_bgr, (input_width, input_height), interpolation=cv2.INTER_AREA)
            boxed_image = self.bounding_box(raw_bgr, res)
            cv2.imshow('res', boxed_image)
            key = cv2.waitKey(0)
            if key == 27:
                break

    def calculate_map(self, dataset='validation', iteration_count=0, save_model=False, device='auto', confidence_threshold=0.2, tp_iou_threshold=0.5, cached=False):
        if dataset == 'train':
            image_paths = self.__train_image_paths
        elif dataset == 'validation':
            image_paths = self.__validation_image_paths
        else:
            print(f'invalid dataset : [{dataset}]')
            return
        device = ModelUtil.available_device() if device == 'auto' else device
        mean_ap, f1_score, iou, tp, fp, fn, confidence = calc_mean_average_precision(
            model=self.__model,
            image_paths=image_paths,
            device=device,
            confidence_threshold=confidence_threshold,
            tp_iou_threshold=tp_iou_threshold,
            classes_txt_path=self.__class_names_file_path,
            cached=cached)
        if save_model:
            model_path = f'{self.__checkpoint_path}/'
            model_path += f'{self.__model_name}'
            if iteration_count + self.__presaved_iteration_count > 0:
                model_path += f'_{iteration_count + self.__presaved_iteration_count}_iter'
            model_path += f'_mAP_{mean_ap:.4f}'
            model_path += f'_f1_{f1_score:.4f}'
            model_path += f'_iou_{iou:.4f}'
            model_path += f'_tp_{tp}_fp_{fp}_fn_{fn}'
            model_path += f'_conf_{confidence:.4f}'
            model_path += f'_confth_{confidence_threshold:.2f}'
            model_path += f'_tpiouth_{tp_iou_threshold:.2f}'
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

