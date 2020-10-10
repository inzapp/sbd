"""
Copyright 2020 inzapp Authors. All Rights Reserved.

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
from time import time

import cv2
import numpy as np
from tensorflow.python.keras import layers, Input, Model
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tqdm import tqdm

from sbd_box_colors import colors

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

img_type = cv2.IMREAD_COLOR
train_img_path = r'.'
test_img_path = r'.'
lr = 0.01
momentum = 0.9
batch_size = 2
epoch = 500
output_shape = (32, 32)
input_shape = (256, 256)
bbox_percentage_threshold = 0.5
bbox_padding_val = 2

font_scale = 0.4
img_channels = 3 if img_type == cv2.IMREAD_COLOR else 1
class_count = 0
class_names = []


def load():
    """
    generate train data for sbd training.
    :return: train_x, train_y
    """
    global train_img_path, class_count, class_names
    with open(rf'{train_img_path}\classes.txt', 'rt') as classes_file:
        class_names = [s.replace('\n', '') for s in classes_file.readlines()]
        class_count = len(class_names)

    img_paths = glob(rf'{train_img_path}\*.jpg') + glob(rf'{train_img_path}\*.png')
    total_x, total_y = [], []
    for cur_img_path in tqdm(img_paths):
        file_name_without_extension = cur_img_path.replace('\\', '/').split('/').pop()[:-4]
        x = cv2.imread(cur_img_path, img_type)
        x = cv2.resize(x, (input_shape[1], input_shape[0]))
        x = np.moveaxis(x, -1, 0)
        total_x.append(x)
        with open(rf'{train_img_path}\{file_name_without_extension}.txt') as file:
            label_lines = file.readlines()
        y = [np.zeros(output_shape, dtype=np.uint8) for _ in range(class_count)]
        for label_line in label_lines:
            class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1, y1, x2, y2 = int(x1 * output_shape[1]), int(y1 * output_shape[0]), int(x2 * output_shape[1]), int(y2 * output_shape[0])
            cv2.rectangle(y[int(class_index)], (x1, y1), (x2, y2), (255, 255, 255), -1)
        total_y.append(y)

    total_x = np.asarray(total_x).reshape(len(total_x), img_channels, input_shape[0], input_shape[1]).astype('float32') / 255.
    total_y = np.asarray(total_y).reshape(len(total_y), class_count, output_shape[0], output_shape[1]).astype('float32') / 255.
    r = np.arange(len(total_x))
    np.random.shuffle(r)
    total_x = total_x[r]
    total_y = total_y[r]
    return total_x, total_y


def predict(model, img):
    """
    detect object in image using trained sbd model.
    :param model: sbd model.
    :param img: image to be predicted.
    :return: dictionary array sorted by x position.
    each dictionary has class index and box info: [x1, y1, x2, y2].
    """
    global bbox_percentage_threshold, bbox_padding_val
    x = cv2.resize(img, (input_shape[1], input_shape[0]))
    x = np.moveaxis(x, -1, 0)
    x = np.array(x).reshape(1, img_channels, input_shape[0], input_shape[1]).astype('float32') / 255
    res_list = model.predict(x=x, batch_size=1)[0]
    predict_res = []
    for i, res in enumerate(res_list):
        res = np.array(res).reshape(output_shape).astype('float32') * 255
        res = cv2.resize(res, (img.shape[1], img.shape[0])).astype('uint8')
        _, res = cv2.threshold(res, int(bbox_percentage_threshold * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = x - bbox_padding_val, y - bbox_padding_val, w + 2 * bbox_padding_val, h + 2 * bbox_padding_val
            predict_res.append({
                'class': i,
                'box': [x, y, x + w, y + h]
            })
    return sorted(predict_res, key=lambda __x: __x['box'][0])


def get_text_label_width_height(text):
    """
    calculate label text position using contour of real text size.
    :param text: label text(class name).
    :return: width, height of label text.
    """
    global font_scale
    black = np.zeros((50, 500), dtype=np.uint8)
    cv2.putText(black, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    black = cv2.resize(black, (0, 0), fx=0.5, fy=0.5)
    black = cv2.dilate(black, np.ones((2, 3), dtype=np.uint8), iterations=2)
    black = cv2.resize(black, (0, 0), fx=2, fy=2)
    _, black = cv2.threshold(black, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    x, y, w, h = cv2.boundingRect(hull)
    return w - 5, h


def is_background_color_bright(bgr):
    """
    determine whether the color is bright or not.
    :param bgr: bgr scalar tuple.
    :return: true if parameter color is bright and false if not.
    """
    tmp = np.zeros((1, 1), dtype=np.uint8)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(tmp, (0, 0), (1, 1), bgr, -1)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    return tmp[0][0] > 127


def bounding_box(img, predict_res):
    """
    draw bounding box using result of sbd.predict function.
    :param img: image to be predicted.
    :param predict_res: result value of sbd.predict() function.
    :return: image of bounding boxed.
    """
    global font_scale
    for i, cur_res in enumerate(predict_res):
        class_index = int(cur_res['class'])
        class_name = class_names[class_index].replace('\n', '')
        label_background_color = colors[class_index]
        label_font_color = (0, 0, 0) if is_background_color_bright(label_background_color) else (255, 255, 255)
        label_width, label_height = get_text_label_width_height(class_name)
        x1, y1, x2, y2 = cur_res['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 2)
        cv2.rectangle(img, (x1 - 1, y1 - label_height), (x1 - 1 + label_width, y1), colors[class_index], -1)
        cv2.putText(img, class_name, (x1 - 1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
    return img


def train():
    """
    train the sbd network using the hyper parameter at the top.
    """
    global lr, momentum, batch_size, epoch, test_img_path, class_names
    total_x, total_y = load()

    model_input = Input(shape=(img_channels, input_shape[0], input_shape[1]))
    x = layers.Conv2D(filters=8, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_first', activation='relu')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(data_format='channels_first')(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_first', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(data_format='channels_first')(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_first', activation='relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_first', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(data_format='channels_first')(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_first', activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_first', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_first', activation='relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_first', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=class_count, kernel_size=(1, 1), padding='same', data_format='channels_first', activation='sigmoid')(x)

    model = Model(model_input, x)
    model.summary()
    model.compile(optimizer=SGD(lr=lr, momentum=momentum), loss='binary_crossentropy')
    model.fit(
        x=total_x,
        y=total_y,
        batch_size=batch_size,
        epochs=epoch
    )

    model.save('sbd.h5')

    for cur_img_path in glob(rf'{test_img_path}\*.jpg') + glob(rf'{test_img_path}\*.png'):
        img = cv2.imread(cur_img_path, img_type)
        res = predict(model, img)
        img = bounding_box(img, res)
        print(res)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    train()
