import os
from time import time

import numpy as np
import tensorflow as tf
from cv2 import cv2

from box_colors import colors
from generator import YoloDataGenerator
from loss import YoloLoss
from metrics import precision, recall, f1, iou_f1
from model import Model


class Yolo:
    def __init__(self, pretrained_model_path='', class_names_file_path=''):
        self.__class_names = []
        self.__input_shape = ()
        self.__model = tf.keras.models.Model()
        self.__train_data_generator = YoloDataGenerator.empty()
        self.__validation_data_generator = YoloDataGenerator.empty()
        self.__live_view_previous_time = time()
        self.__callbacks = [
            tf.keras.callbacks.LambdaCallback(on_batch_end=self.__training_view),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/epoch_{epoch}_f1_{f1:.4f}_val_f1_{val_f1:.4f}.h5',
                monitor='val_f1',
                mode='max',
                save_best_only=True)]

        # TODO : 1. 모델 로드하지 않음 -> 훈련, 2. 모델 로드 -> 이어서 훈련, 3. 모델 로드 -> predict
        # TODO : 이 3가지가 서로 간섭받지 않아야 하며 깔끔하게 모듈화가 되어야 한다.
        # TODO : 어떻게 할 것인가.

        if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
            self.__class_names, _ = self.__init_class_names(class_names_file_path)
            self.__model = tf.keras.models.load_model(pretrained_model_path, compile=False)

    def fit(self, train_image_path, input_shape, batch_size, lr, epochs, validation_split=0.0, validation_image_path=''):
        num_classes = 0
        self.__input_shape = input_shape
        if len(self.__class_names) == 0:
            self.__class_names, num_classes = self.__init_class_names(f'{train_image_path}/classes.txt')
        if len(self.__model.layers) == 0:
            self.__model = Model(input_shape, num_classes + 5).build()
        self.__model.summary()
        self.__model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss=YoloLoss(),
            metrics=[precision, recall, f1],
            run_eagerly=True)

        self.__train_data_generator = YoloDataGenerator(
            train_image_path=train_image_path,
            input_shape=input_shape,
            output_shape=self.__model.output_shape[1:],
            batch_size=batch_size,
            validation_split=validation_split)
        print(f'\ntrain on {len(self.__train_data_generator.train_image_paths)} samples.')
        if os.path.exists(validation_image_path) and os.path.isdir(validation_image_path):
            self.__validation_data_generator = YoloDataGenerator(
                train_image_path=validation_image_path,
                input_shape=input_shape,
                output_shape=self.__model.output_shape[1:],
                batch_size=batch_size)
            print(f'validate on {len(self.__validation_data_generator.train_image_paths)} samples.')
            self.__model.fit(
                x=self.__train_data_generator.flow(),
                validation_data=self.__validation_data_generator.flow(),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=self.__callbacks)
        elif len(self.__train_data_generator.validation_image_paths) > 0:
            print(f'validate on {len(self.__train_data_generator.validation_image_paths)} samples.')
            self.__model.fit(
                x=self.__train_data_generator.flow('training'),
                validation_data=self.__train_data_generator.flow('validation'),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=self.__callbacks)
        else:
            self.__model.fit(
                x=self.__train_data_generator.flow(),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=self.__callbacks)

    def predict(self, x, confidence_threshold=0.25, nms_iou_threshold=0.5):
        """
        Detect object in image using trained YOLO model.
        :param x: image to be predicted.
        :param confidence_threshold: threshold confidence score to detect object.
        :param nms_iou_threshold: threshold to remove overlapped detection.
        :return: dictionary array sorted by x position.
        each dictionary has class index and bbox info: [x1, y1, x2, y2].
        """
        raw_width, raw_height = x.shape[1], x.shape[0]
        input_shape = self.__model.input.shape[1:]
        output_shape = self.__model.output.shape[1:]
        if x.shape[1] > input_shape[1] or x.shape[0] > input_shape[0]:
            x = cv2.resize(x, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        else:
            x = cv2.resize(x, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)

        x = np.asarray(x).reshape((1,) + input_shape).astype('float32') / 255.0
        y = self.__predict_on_graph(self.__model, x)[0]
        y = np.moveaxis(y, -1, 0)

        res = []
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                confidence = y[0][i][j]
                if confidence < confidence_threshold:
                    continue
                cx = y[1][i][j]
                cx_f = j / float(output_shape[1])
                cx_f += 1 / float(output_shape[1]) * cx
                cy = y[2][i][j]
                cy_f = i / float(output_shape[0])
                cy_f += 1 / float(output_shape[0]) * cy
                w = y[3][i][j]
                h = y[4][i][j]

                x_min_f = cx_f - w / 2.0
                y_min_f = cy_f - h / 2.0
                x_max_f = cx_f + w / 2.0
                y_max_f = cy_f + h / 2.0
                x_min = int(x_min_f * raw_width)
                y_min = int(y_min_f * raw_height)
                x_max = int(x_max_f * raw_width)
                y_max = int(y_max_f * raw_height)
                class_index = -1
                max_percentage = -1
                for cur_channel_index in range(5, len(y)):
                    if max_percentage < y[cur_channel_index][i][j]:
                        class_index = cur_channel_index
                        max_percentage = y[cur_channel_index][i][j]
                res.append({
                    'confidence': confidence,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'class': class_index - 5,
                    'discard': False})

        for i in range(len(res)):
            if res[i]['discard']:
                continue
            for j in range(len(res)):
                if i == j or res[j]['discard']:
                    continue
                if self.__iou(res[i]['bbox'], res[j]['bbox']) > nms_iou_threshold:
                    if res[i]['confidence'] >= res[j]['confidence']:
                        res[j]['discard'] = True

        res_copy = np.asarray(res.copy())
        res = []
        for i in range(len(res_copy)):
            if not res_copy[i]['discard']:
                res.append(res_copy[i])
        return sorted(res, key=lambda __x: __x['bbox'][0])

    def bounding_box(self, img, yolo_res, font_scale=0.4):
        """
        draw bounding bbox using result of YOLO.predict function.
        :param img: image to be predicted.
        :param yolo_res: result value of YOLO.predict() function.
        :param font_scale: scale of font.
        :return: image of bounding boxed.
        """
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, cur_res in enumerate(yolo_res):
            class_index = int(cur_res['class'])
            if len(self.__class_names) == 0:
                class_name = str(class_index)
            else:
                class_name = self.__class_names[class_index].replace('/n', '')
            label_background_color = colors[class_index]
            label_font_color = (0, 0, 0) if self.__is_background_color_bright(label_background_color) else (255, 255, 255)
            label_text = f'{class_name}({round(cur_res["confidence"] * 100.0)}%)'
            label_width, label_height = self.__get_text_label_width_height(label_text, font_scale)
            x1, y1, x2, y2 = cur_res['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 2)
            cv2.rectangle(img, (x1 - 1, y1 - label_height), (x1 - 1 + label_width, y1), colors[class_index], -1)
            cv2.putText(img, label_text, (x1 - 1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
        return img

    def evaluate(self):
        if len(self.__train_data_generator.validation_image_paths) > 0:
            evaluate_image_paths = self.__train_data_generator.validation_image_paths
        elif len(self.__validation_data_generator.train_image_paths) > 0:
            evaluate_image_paths = self.__validation_data_generator.train_image_paths
        else:
            print('no validation set specified. evaluate on training set.')
            evaluate_image_paths = self.__train_data_generator.train_image_paths
        for path in evaluate_image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.__model.input.shape[-1] == 1 else cv2.IMREAD_COLOR)
            res = self.predict(img)
            boxed_image = self.bounding_box(img, res)
            cv2.imshow('res', boxed_image)
            cv2.waitKey(0)

    def evaluate_video(self, video_path):
        pass

    def __training_view(self, batch, logs):
        cur_time = time()
        if cur_time - self.__live_view_previous_time > 0.5:
            self.__live_view_previous_time = cur_time
            index = np.random.randint(0, len(self.__train_data_generator.train_image_paths))
            img_path = self.__train_data_generator.train_image_paths[index]
            if len(self.__train_data_generator.validation_image_paths) > 0:
                if np.random.choice([0, 1]) == 1:
                    index = np.random.randint(0, len(self.__train_data_generator.validation_image_paths))
                    img_path = self.__train_data_generator.validation_image_paths[index]
            elif len(self.__validation_data_generator.train_image_paths) > 0:
                if np.random.choice([0, 1]) == 1:
                    index = np.random.randint(0, len(self.__validation_data_generator.train_image_paths))
                    img_path = self.__validation_data_generator.train_image_paths[index]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if self.__model.input.shape[-1] == 1 else cv2.IMREAD_COLOR)
            res = self.predict(img)
            boxed_image = self.bounding_box(img, res)
            cv2.imshow('training view', boxed_image)
            cv2.waitKey(1)

    def __iou_f1(self):
        # calculate on training set
        for image_path in self.__train_data_generator.train_image_paths:
            label_path = f'{image_path[:-4]}.txt'
            with open(label_path, mode='rt') as f:
                label_lines = f.readlines()
            bbox_true = []
            for line in label_lines:
                class_index, cx, cy, w, h = list(map(float, line.split(' ')))
                x1, x2 = cx - w / 2.0, cx + w / 2.0
                y1, y2 = cy - h / 2.0, cy + h / 2.0
                x1, x2 = int(x1 * self.__model.input_shape[1:][1]), int(x2 * self.__model.input_shape[1:][1])
                y1, y2 = int(y1 * self.__model.input_shape[1:][0]), int(y2 * self.__model.input_shape[1:][0])
                bbox_true.append([int(class_index), x1, y1, x2, y2])

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.__model.input.input_shape[-1] == 1 else cv2.IMREAD_COLOR)
            res = self.predict(img)
            bbox_pred = []
            for cur_res in res:
                bbox_pred.append([cur_res['class']] + cur_res['bbox'])

        # calculate on validation set
        if len(self.__train_data_generator.validation_image_paths) > 0:
            validation_image_paths = self.__train_data_generator.validation_image_paths
        elif len(self.__validation_data_generator.train_image_paths) > 0:
            validation_image_paths = self.__validation_data_generator.train_image_paths
        else:
            return
        for image_path in validation_image_paths:
            pass

    @tf.function
    def __predict_on_graph(self, model, x):
        return model(x, training=False)

    @staticmethod
    def __init_class_names(class_names_file_path):
        """
        Init YOLO label from classes.txt file.
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
        if intersection_width < 0.0 or intersection_height < 0.0:
            return 0.0
        intersection_area = intersection_width * intersection_height
        a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
        b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
        union_area = a_area + b_area - intersection_area
        return intersection_area / float(union_area)

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

    @staticmethod
    def __get_text_label_width_height(text, font_scale):
        """
        Calculate label text position using contour of real text size.
        :param text: label text(class name).
        :param font_scale: scale of font.
        :return: width, height of label text.
        """
        black = np.zeros((50, 500), dtype=np.uint8)
        cv2.putText(black, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        black = cv2.resize(black, (int(black.shape[1] / 2), int(black.shape[0] / 2)), interpolation=cv2.INTER_AREA)
        black = cv2.dilate(black, np.ones((2, 2), dtype=np.uint8), iterations=2)
        black = cv2.resize(black, (int(black.shape[1] * 2), int(black.shape[0] * 2)), interpolation=cv2.INTER_LINEAR)
        _, black = cv2.threshold(black, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])
        x, y, w, h = cv2.boundingRect(hull)
        return w - 5, h
