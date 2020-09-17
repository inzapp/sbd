# Copyright 2020 inzapp Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"),
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from glob import glob

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
alpha = 0.0
lr = 0.01
momentum = 0.9
batch_size = 2
epoch = 500
grid = (32, 32)
size = (256, 256)
bbox_percentage_threshold = 0.5
bbox_padding_val = 10

font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.5
thickness = 1

grid_width = int(size[0] / grid[0])
grid_height = int(size[1] / grid[1])
img_channels = 3 if img_type == cv2.IMREAD_COLOR else 1
class_count = 0
class_names = []
model = None


def load():
    global train_img_path, class_count, class_names
    with open(rf'{train_img_path}\classes.txt', 'rt') as classes_file:
        class_names = classes_file.readlines()
        class_count = len(class_names)

    img_paths = glob(rf'{train_img_path}\*.jpg') + glob(rf'{train_img_path}\*.png')
    total_x, total_y = [], [[] for _ in range(class_count)]
    for cur_img_path in tqdm(img_paths):
        file_name_without_extension = cur_img_path.replace('\\', '/').split('/').pop().split('.')[0]
        x = cv2.imread(cur_img_path, img_type)
        x = cv2.resize(x, size)
        total_x.append(x)
        with open(rf'{train_img_path}\{file_name_without_extension}.txt') as file:
            masks = [np.zeros(size[0] * size[1]).reshape(size[0], size[1]) for _ in range(class_count)]
            for line in file.readlines():
                class_index, cx, cy, w, h = np.array(line.split(' ')).astype('float32')
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                x1, y1, x2, y2 = int(x1 * size[0]), int(y1 * size[1]), int(x2 * size[0]), int(y2 * size[1])
                cv2.rectangle(masks[int(class_index)], (x1, y1), (x2, y2), (255, 255, 255), -1)
            for mask_index, _ in enumerate(masks):
                segmentation_output = []
                for col in range(0, size[0], grid_width):
                    for row in range(0, size[1], grid_height):
                        cur_grid = masks[mask_index][col:col + grid_height, row:row + grid_width]
                        non_zero_count = cv2.countNonZero(cur_grid)
                        cur_grid_segmentation_ratio = non_zero_count / (grid_width * grid_height)
                        segmentation_output.append(cur_grid_segmentation_ratio)
                total_y[mask_index].append(segmentation_output)

    total_x = np.asarray(total_x).reshape(len(total_x), size[0], size[1], img_channels).astype('float32') / 255
    total_y = np.asarray(total_y).reshape(class_count, len(total_y[0]), grid[0], grid[1]).astype('float32')
    r = np.arange(len(total_x))
    np.random.shuffle(r)
    total_x = total_x[r]
    new_total_y = []
    for i in range(len(total_y)):
        cur_total_y = total_y[i]
        cur_total_y = cur_total_y[r]
        new_total_y.append(cur_total_y)
    return total_x, new_total_y


def epoch_end_multiple_output_callback(cur_epoch, logs):
    global class_count
    min_loss, max_loss, sum_of_avg_loss = 999999999, 0, 0
    print(f'epoch {cur_epoch}')
    for i in range(class_count):
        cur_loss = logs[f'r{i}_loss']
        min_loss = cur_loss if cur_loss < min_loss else min_loss
        max_loss = cur_loss if cur_loss > max_loss else max_loss
        sum_of_avg_loss += cur_loss
    avg_loss = sum_of_avg_loss / class_count
    print(f'min_loss : {min_loss:.4f}, max_loss : {max_loss:.4f}, avg_loss : {avg_loss:.4f}')
    min_loss, max_loss, sum_of_avg_loss = 999999999, 0, 0
    for i in range(class_count):
        cur_loss = logs[f'val_r{i}_loss']
        min_loss = cur_loss if cur_loss < min_loss else min_loss
        max_loss = cur_loss if cur_loss > max_loss else max_loss
        sum_of_avg_loss += cur_loss
    avg_loss = sum_of_avg_loss / class_count
    print(f'min_val_loss : {min_loss:.4f}, max_val_loss : {max_loss:.4f}, avg_val_loss : {avg_loss:.4f}')
    print()


def predict(img):
    global model, bbox_percentage_threshold, bbox_padding_val
    x = cv2.resize(img, size)
    x = np.array(x).reshape(1, size[0], size[1], img_channels).astype('float32') / 255
    res_list = model.predict(x=x, batch_size=1)
    predict_res = []
    for i, res in enumerate(res_list):
        res = np.array(res).reshape(grid[0], grid[1], 1).astype('float32') * 255
        res = cv2.resize(res, (img.shape[1], img.shape[0])).astype('uint8')
        _, res = cv2.threshold(res, int(bbox_percentage_threshold * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = x - bbox_padding_val, y - bbox_padding_val, w + 2 * bbox_padding_val, h + 2 * bbox_padding_val
            predict_res.append({
                'class': i,
                'box': [x, y, x + w, y + h]
            })
    return sorted(predict_res, key=lambda __x: __x['box'][0])


def get_text_label_width_height(text):
    global font, font_scale, thickness
    black = np.zeros(50 * 300).reshape(50, 300).astype('uint8')
    cv2.putText(black, text, (30, 30), font, fontScale=font_scale, color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    black = cv2.resize(black, (0, 0), fx=0.5, fy=0.5)
    black = cv2.dilate(black, np.ones((2, 3), dtype=np.uint8), iterations=2)
    black = cv2.resize(black, (0, 0), fx=2, fy=2)
    _, black = cv2.threshold(black, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    x, y, w, h = cv2.boundingRect(hull)
    return w - 5, h


def is_background_color_bright(bgr):
    tmp = np.zeros((1, 1), dtype=np.uint8)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(tmp, (0, 0), (1, 1), bgr, -1)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    return tmp[0][0] > 127


def bounding_box(img, predict_res):
    global font, font_scale, thickness
    for i, cur_res in enumerate(predict_res):
        class_index = int(cur_res['class'])
        class_name = class_names[class_index].replace('\n', '')
        label_background_color = colors[class_index]
        label_font_color = (0, 0, 0) if is_background_color_bright(label_background_color) else (255, 255, 255)
        label_width, label_height = get_text_label_width_height(class_name)
        x1, y1, x2, y2 = cur_res['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 2)
        cv2.rectangle(img, (x1 - 1, y1 - label_height), (x1 - 1 + label_width, y1), colors[class_index], -1)
        cv2.putText(img, class_name, (x1 - 1, y1 - 5), font, fontScale=font_scale, color=label_font_color, thickness=thickness, lineType=cv2.LINE_AA)
    return img


def train():
    global alpha, lr, momentum, batch_size, epoch, test_img_path, model
    total_x, total_y = load()

    model_input = Input(shape=(size[0], size[1], img_channels))

    x = layers.Conv2D(filters=4, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(filters=8, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=alpha)(x)

    model_outputs = []
    for i in range(class_count):
        model_output = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
        model_output = layers.Reshape(target_shape=(grid[0], grid[1]), name=f'r{i}')(model_output)
        model_outputs.append(model_output)

    model = Model(model_input, model_outputs)

    model.summary()
    model.compile(optimizer=SGD(lr=lr, momentum=momentum, nesterov=True), loss='binary_crossentropy')
    model.fit(
        x=total_x,
        y=total_y,
        batch_size=batch_size,
        epochs=epoch
    )

    model.save('sbd.h5')

    for cur_img_path in glob(rf'{test_img_path}\*.jpg') + glob(rf'{test_img_path}\*.png'):
        img = cv2.imread(cur_img_path, img_type)
        res = predict(img)
        img = bounding_box(img, res)
        print(res)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    train()
