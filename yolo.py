import os

import numpy as np
import tensorflow as tf
from cv2 import cv2

from box_colors import colors
from generator import YoloDataGenerator
from loss import YoloLoss
from model import Model


class Yolo:
    def __init__(self, pretrained_model_path='', class_names_file_path=''):
        self._class_names = []
        self._model = tf.keras.models.Model()
        self._callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/epoch_{epoch}.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True)]

        # TODO : 1. 모델 로드하지 않음 -> 훈련, 2. 모델 로드 -> 이어서 훈련, 3. 모델 로드 -> predict
        # TODO : 이 3가지가 서로 간섭받지 않아야 하며 깔끔하게 모듈화가 되어야 한다.
        # TODO : 어떻게 할 것인가.

        if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
            self._class_names, _ = self._init_class_names(class_names_file_path)
            self._model = tf.keras.models.load_model(pretrained_model_path, compile=False)

    def fit(self, train_image_path, input_shape, batch_size, lr, epochs, validation_split=0.0, validation_image_path=''):
        num_classes = 0
        if len(self._class_names) == 0:
            self._class_names, num_classes = self._init_class_names(f'{train_image_path}/classes.txt')
        if len(self._model.layers) == 0:
            self._model = Model(input_shape, num_classes + 5).build()
        self._model.summary()
        self._model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True),
            loss=YoloLoss(num_classes))
        train_data_generator = YoloDataGenerator(
            train_image_path=train_image_path,
            input_shape=input_shape,
            output_shape=self._model.output_shape[1:],
            batch_size=batch_size,
            validation_split=validation_split)
        if os.path.exists(validation_image_path) and os.path.isdir(validation_image_path):
            validation_data_generator = YoloDataGenerator(
                train_image_path=validation_image_path,
                input_shape=input_shape,
                output_shape=self._model.output_shape[1:],
                batch_size=batch_size)
            self._model.fit(
                x=train_data_generator.flow(),
                validation_data=validation_data_generator.flow(),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=self._callbacks)
        else:
            self._model.fit(
                x=train_data_generator.flow('training'),
                validation_data=train_data_generator.flow('validation'),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=self._callbacks)

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
        input_shape = self._model.input.shape[1:]
        output_shape = self._model.output.shape[1:]
        if x.shape[1] > input_shape[1] or x.shape[0] > input_shape[0]:
            x = cv2.resize(x, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        else:
            x = cv2.resize(x, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)

        x = np.asarray(x).reshape((1,) + input_shape).astype('float32') / 255.0
        y = self._predict_on_graph(self._model, x)
        y = np.moveaxis(y, -1, 0)

        res = []
        # box_count = 0
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
            #     box_count += 1
            #     if box_count >= max_num_boxes:
            #         break
            # if box_count >= max_num_boxes:
            #     break

        # nms process
        for i in range(len(res)):
            if res[i]['discard']:
                continue
            for j in range(len(res)):
                if i == j or res[j]['discard']:
                    continue
                if self._iou(res[i]['bbox'], res[j]['bbox']) > nms_iou_threshold:
                    if res[i]['confidence'] >= res[j]['confidence']:
                        res[j]['discard'] = True

        res_copy = res.copy()
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
            if len(self._class_names) == 0:
                class_name = str(class_index)
            else:
                class_name = self._class_names[class_index].replace('/n', '')
            label_background_color = colors[class_index]
            label_font_color = (0, 0, 0) if self._is_background_color_bright(label_background_color) else (255, 255, 255)
            label_text = f'{class_name}({round(cur_res["confidence"] * 100.0)}%)'
            label_width, label_height = self._get_text_label_width_height(label_text, font_scale)
            x1, y1, x2, y2 = cur_res['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 2)
            cv2.rectangle(img, (x1 - 1, y1 - label_height), (x1 - 1 + label_width, y1), colors[class_index], -1)
            cv2.putText(img, label_text, (x1 - 1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
        return img

    @tf.function
    def _predict_on_graph(self, model, x):
        return model(x, training=False)

    @staticmethod
    def _init_class_names(class_names_file_path):
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
    def _iou(a, b):
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
    def _is_background_color_bright(bgr):
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
    def _get_text_label_width_height(text, font_scale):
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
