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
import random
from glob import glob
from time import time

import cv2
import numpy as np
import tensorflow as tf

from sbd_box_colors import colors

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

img_type = cv2.IMREAD_GRAYSCALE
train_image_path = r'.'
test_img_path = r'.'

lr = 0.01
batch_size = 2
epoch = 20
validation_ratio = 0.2
input_shape = (256, 256)
output_shape = (32, 32)
bbox_percentage_threshold = 0.25
bbox_padding_val = 0

font_scale = 0.4
img_channels = 3 if img_type == cv2.IMREAD_COLOR else 1
class_count = 0
class_names = []

new_model_saved = True


class SbdDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for SBD model.
    Usage:
        generator = SbdDataGenerator(image_paths=train_image_paths, augmentation=True)
    """

    def __init__(self, image_paths, augmentation):
        self.init_label()
        self.image_paths = image_paths
        self.augmentation = augmentation
        self.random_indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.random_indexes)

    def __getitem__(self, index):
        global img_type, batch_size, input_shape, img_channels, class_count
        batch_x = []
        batch_y = []
        start_index = index * batch_size
        for i in range(start_index, start_index + batch_size):
            cur_img_path = self.image_paths[self.random_indexes[i]]
            x = cv2.imread(cur_img_path, img_type)
            if x.shape[1] > input_shape[1] or x.shape[0] > input_shape[0]:
                x = cv2.resize(x, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
            else:
                x = cv2.resize(x, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
            with open(f'{cur_img_path[:-4]}.txt', mode='rt') as file:
                label_lines = file.readlines()
            y = [np.zeros(input_shape, dtype=np.uint8) for _ in range(class_count)]
            for label_line in label_lines:
                class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                x1, y1, x2, y2 = int(x1 * input_shape[1]), int(y1 * input_shape[0]), int(x2 * input_shape[1]), int(y2 * input_shape[0])
                cv2.rectangle(y[int(class_index)], (x1, y1), (x2, y2), (255, 255, 255), -1)

            if self.augmentation:
                if random.choice([0, 1]) == 1:
                    if img_channels == 1:
                        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
                    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
                    hsv = np.asarray(hsv).astype('float32')
                    hsv = np.moveaxis(hsv, -1, 0)
                    if random.choice([0, 1]) == 1:
                        hsv[1] *= random.uniform(0.25, 1.75)
                    if random.choice([0, 1]) == 1:
                        hsv[2] *= random.uniform(0.75, 1.25)
                    hsv = np.moveaxis(hsv, 0, -1)
                    hsv = np.clip(hsv, 0, 255).astype('uint8')
                    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    x = np.asarray(x).astype('float32')
                    x = np.clip(x, 0, 255).astype('uint8')
                    if img_channels == 1:
                        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

                if random.choice([0, 1]) == 1:
                    top_padding = random.randrange(0, int(input_shape[0] * 0.15), 1)
                    bottom_padding = random.randrange(0, int(input_shape[0] * 0.15), 1)
                    left_padding = random.randrange(0, int(input_shape[1] * 0.15), 1)
                    right_padding = random.randrange(0, int(input_shape[1] * 0.15), 1)
                    x = cv2.copyMakeBorder(
                        src=x,
                        top=top_padding,
                        bottom=bottom_padding,
                        left=left_padding,
                        right=right_padding,
                        borderType=cv2.BORDER_CONSTANT,
                        value=0
                    )
                    x = cv2.resize(x, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
                    for j in range(len(y)):
                        y[j] = cv2.copyMakeBorder(
                            src=y[j],
                            top=top_padding,
                            bottom=bottom_padding,
                            left=left_padding,
                            right=right_padding,
                            borderType=cv2.BORDER_CONSTANT,
                            value=0
                        )

            for j in range(len(y)):
                y[j] = self.compress(y[j])
            x = np.asarray(x).reshape(input_shape[0], input_shape[1], img_channels).astype('float32') / 255.
            y = np.moveaxis(np.asarray(y), 0, -1)
            y = np.asarray(y).reshape(output_shape[0], output_shape[1], class_count).astype('float32')
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        return batch_x, batch_y

    def __len__(self):
        global batch_size
        return int(np.floor(len(self.image_paths) / batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.random_indexes)

    @staticmethod
    def compress(y):
        """
        Compress sbd label to between 0 or 1.
        :param y: masked sbd label to be compressed.
        """
        global input_shape, output_shape
        assert input_shape[1] % output_shape[1] == 0
        assert input_shape[0] % output_shape[0] == 0
        grid_width = int(input_shape[1] / output_shape[1])
        grid_height = int(input_shape[0] / output_shape[0])
        grid_area = float(grid_width * grid_height)
        compressed_y = []
        for grid_y in range(0, input_shape[0], grid_height):
            row = []
            for grid_x in range(0, input_shape[1], grid_width):
                grid = y[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]
                score = cv2.countNonZero(grid) / grid_area
                row.append(score)
            compressed_y.append(row)
        return np.asarray(compressed_y)

    @staticmethod
    def init_label():
        """
        Init sbd label from classes.txt file.
        """
        global train_image_path, class_count, class_names
        if class_count == 0:
            with open(f'{train_image_path}/classes.txt', 'rt') as classes_file:
                class_names = [s.replace('\n', '') for s in classes_file.readlines()]
                class_count = len(class_names)


class FPWeightedError(tf.keras.losses.Loss):
    """
    False positive weighted loss function.
    f(x) = -log(1 - MAE(x)) + -log(1 - FP(x))
    Usage:
     model.compile(loss=[FPWeightedError()], optimizer="sgd")
    """

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = tf.math.abs(y_pred - y_true)
        loss = -tf.math.log(1.0 + 1e-7 - loss)
        loss = tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
        fp = tf.keras.backend.clip(-(y_true - y_pred), 0.0, 1.0)
        fp = -tf.math.log(1.0 + 1e-7 - fp)
        fp = tf.keras.backend.mean(tf.keras.backend.sum(fp, axis=-1))
        return loss + fp


def resize(img, size):
    """
    Use different interpolations to resize according to the target size.
    :param img: image to be resized.
    :param size: target size (width, height).
    """
    if img.shape[1] > size[0] or img.shape[0] > size[1]:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def forward(net, x):
    """
    Detect object in image using trained sbd model.
    :param net: sbd tensorflow frozen pb model.
    :param x: image to be predicted.
    :return: dictionary array sorted by x position.
    each dictionary has class index and box info: [x1, y1, x2, y2].
    """
    global bbox_percentage_threshold, bbox_padding_val, class_names
    raw_width, raw_height = x.shape[1], x.shape[0]
    if img_channels == 1:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = resize(x, (input_shape[1], input_shape[0]))
    x = np.asarray(x).reshape(1, img_channels, input_shape[0], input_shape[1]).astype('float32') / 255.
    net.setInput(x)
    y = net.forward()[0]
    predict_res = []

    for i, channel in enumerate(y):
        channel = np.asarray(channel).reshape(output_shape).astype('float32') * 255
        channel = cv2.resize(channel, (raw_width, raw_height), interpolation=cv2.INTER_LINEAR).astype('uint8')
        _, channel = cv2.threshold(channel, int(bbox_percentage_threshold * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    Calculate label text position using contour of real text size.
    :param text: label text(class name).
    :return: width, height of label text.
    """
    global font_scale
    black = np.zeros((50, 500), dtype=np.uint8)
    cv2.putText(black, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    black = resize(black, (int(black.shape[1] / 2), int(black.shape[0] / 2)))
    black = cv2.dilate(black, np.ones((2, 2), dtype=np.uint8), iterations=2)
    black = resize(black, (int(black.shape[1] * 2), int(black.shape[0] * 2)))
    _, black = cv2.threshold(black, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    x, y, w, h = cv2.boundingRect(hull)
    return w - 5, h


def is_background_color_bright(bgr):
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


def bounding_box(img, predict_res):
    """
    draw bounding box using result of sbd.predict function.
    :param img: image to be predicted.
    :param predict_res: result value of sbd.predict() function.
    :return: image of bounding boxed.
    """
    global font_scale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, cur_res in enumerate(predict_res):
        class_index = int(cur_res['class'])
        class_name = class_names[class_index].replace('/n', '')
        label_background_color = colors[class_index]
        label_font_color = (0, 0, 0) if is_background_color_bright(label_background_color) else (255, 255, 255)
        label_width, label_height = get_text_label_width_height(class_name)
        x1, y1, x2, y2 = cur_res['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 2)
        cv2.rectangle(img, (x1 - 1, y1 - label_height), (x1 - 1 + label_width, y1), colors[class_index], -1)
        cv2.putText(img, class_name, (x1 - 1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
    return img


def live_view(image_paths):
    """
    Check the SBD models training course by forwarding it in real time.
    :param image_paths: Image paths to be viewed in real time.
    """

    def thread_function():
        """
        Thread function of live_view function.
        """
        global img_type, new_model_saved
        freeze('model.h5')
        net = cv2.dnn.readNet('model.pb')
        while True:
            for cur_image_path in image_paths:
                if new_model_saved:
                    freeze('model.h5')
                    net = cv2.dnn.readNet('model.pb')
                    new_model_saved = False
                img = cv2.imread(cur_image_path, cv2.IMREAD_COLOR)
                img = resize(img, (input_shape[1], input_shape[0]))
                res = forward(net, img)
                img = bounding_box(img, res)
                cv2.imshow('training view', img)
                cv2.waitKey(200)

    from concurrent.futures.thread import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)
    pool.submit(thread_function)


def new_model_saved_on(_epoch, _logs):
    """
    Lambda callback function for alerting new model is saved.
    Usage:
        tf.keras.callbacks.LambdaCallback(on_epoch_end=new_model_saved_on)
    """
    global new_model_saved
    new_model_saved = True


def freeze(model_name):
    """
    Freeze keras h5 model to tensorflow pb file
    :param model_name: keras h5 model name to be frozen.
    """
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model = tf.keras.models.load_model(model_name, compile=False)
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=".",
        name="model.pb",
        as_text=False
    )


def train():
    """
    train the sbd network using the hyper parameter at the top.
    """
    global lr, batch_size, epoch, test_img_path, class_names, class_count, validation_ratio, new_model_saved

    total_image_paths = glob(f'{train_image_path}/*.jpg')
    random.shuffle(total_image_paths)
    train_image_count = int(len(total_image_paths) * (1 - validation_ratio))
    train_image_paths = total_image_paths[:train_image_count]
    validation_image_paths = total_image_paths[train_image_count:]
    train_data_generator = SbdDataGenerator(image_paths=train_image_paths, augmentation=False)
    validation_data_generator = SbdDataGenerator(image_paths=validation_image_paths, augmentation=False)

    print(f'train image count : {len(train_image_paths)}')
    print(f'validation image count : {len(validation_image_paths)}')

    model_input = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], img_channels))

    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same'
    )(model_input)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same'
    )(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same'
    )(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same'
    )(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same'
    )(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters=class_count,
        kernel_size=1,
        kernel_initializer='glorot_uniform',
        activation='sigmoid'
    )(x)

    model = tf.keras.models.Model(model_input, x)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr), loss=FPWeightedError())
    model.save('model.h5')

    live_view(total_image_paths)
    model.fit(
        x=train_data_generator,
        validation_data=validation_data_generator,
        epochs=epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath='model.h5'),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=new_model_saved_on),
        ]
    )
    model.save('model.h5')

    freeze('model.h5')
    net = cv2.dnn.readNet('model.pb')
    for cur_img_path in glob(f'{test_img_path}/*.jpg'):
        print(cur_img_path)
        img = cv2.imread(cur_img_path, cv2.IMREAD_COLOR)
        img = resize(img, (input_shape[1], input_shape[0]))
        st = time()
        res = forward(net, img)
        et = time()
        print(f'[Inference Time] : {(et - st):.3f} s')
        img = bounding_box(img, res)
        print(res)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    train()
