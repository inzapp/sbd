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

from yolo_box_color import colors

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

img_type = cv2.IMREAD_GRAYSCALE
train_image_path = r'C:\inz\train_data\lp_detection'
test_img_path = r'C:\inz\train_data\lp_detection'

lr = 1e-2
batch_size = 2
epoch = 300
validation_ratio = 0.2
input_shape = (368, 640)
output_shape = (23, 40)
bbox_percentage_threshold = 0.25
bbox_padding_val = 0

font_scale = 0.4
img_channels = 3 if img_type == cv2.IMREAD_COLOR else 1
class_count = 0
class_names = []

new_model_saved = True


class YoloDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for YOLO model.
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
            y = [np.zeros(input_shape, dtype=np.uint8) for _ in range(class_count + 5)]
            for label_line in label_lines:
                class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                x1, y1, x2, y2 = int(x1 * input_shape[1]), int(y1 * input_shape[0]), int(x2 * input_shape[1]), int(y2 * input_shape[0])
                cv2.rectangle(y[int(class_index + 5)], (x1, y1), (x2, y2), (255, 255, 255), -1)
            for j in range(len(y)):
                y[j] = self.compress(y[j])
            grid_width_ratio = 1 / float(output_shape[1])
            grid_height_ratio = 1 / float(output_shape[0])
            for label_line in label_lines:
                class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
                center_row = int(cy * output_shape[0])
                center_col = int(cx * output_shape[1])
                y[0][center_row][center_col] = 1.0
                y[1][center_row][center_col] = (cx - (center_col * grid_width_ratio)) / grid_width_ratio
                y[2][center_row][center_col] = (cy - (center_row * grid_height_ratio)) / grid_height_ratio
                y[3][center_row][center_col] = w
                y[4][center_row][center_col] = h
            x = np.asarray(x).reshape(input_shape[0], input_shape[1], img_channels).astype('float32') / 255.
            y = np.moveaxis(np.asarray(y), 0, -1)
            y = np.asarray(y).reshape(output_shape[0], output_shape[1], class_count + 5).astype('float32')
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
        Compress YOLO label to between 0 or 1.
        :param y: masked YOLO label to be compressed.
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
        Init YOLO label from classes.txt file.
        """
        global train_image_path, class_count, class_names
        if class_count == 0:
            with open(f'{train_image_path}/classes.txt', 'rt') as classes_file:
                class_names = [s.replace('\n', '') for s in classes_file.readlines()]
                class_count = len(class_names)


class MeanAbsoluteLogError(tf.keras.losses.Loss):
    """
    Mean absolute log loss function.
    f(x) = -log(1 - MAE(x))
    Usage:
     model.compile(loss=[MeanAbsoluteLogError()], optimizer="sgd")
    """

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = tf.math.abs(y_true - y_pred)
        loss = -tf.math.log(1.0 + 1e-7 - loss)
        loss = tf.keras.backend.mean(loss, axis=-1)
        return loss


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


def forward_yolo(net, x):
    """
    Detect object in image using trained YOLO model.
    :param net: YOLO tensorflow frozen pb model.
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

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            if y[0][i][j] < 0.5:
                continue
            p = y[0][i][j]
            cx = y[1][i][j]
            cx_f = j / float(output_shape[1])
            cx_f += 1 / float(output_shape[1]) * cx
            cy = y[2][i][j]
            cy_f = i / float(output_shape[0])
            cy_f += 1 / float(output_shape[0]) * cy
            w = y[3][i][j]
            h = y[4][i][j]

            x_min_f = cx_f - w / 2
            y_min_f = cy_f - h / 2
            x_max_f = cx_f + w / 2
            y_max_f = cy_f + h / 2
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
            predict_res.append({
                'class': class_index - 5,
                'box': [x_min, y_min, x_max, y_max],
                'p': p
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
    draw bounding box using result of YOLO.predict function.
    :param img: image to be predicted.
    :param predict_res: result value of YOLO.predict() function.
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
        label_text = f'{class_name} {int(cur_res["p"] * 100)}%'
        label_width, label_height = get_text_label_width_height(label_text)
        x1, y1, x2, y2 = cur_res['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), label_background_color, 2)
        cv2.rectangle(img, (x1 - 1, y1 - label_height), (x1 - 1 + label_width, y1), colors[class_index], -1)
        cv2.putText(img, label_text, (x1 - 1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=label_font_color, thickness=1, lineType=cv2.LINE_AA)
    return img


def live_view(image_paths):
    """
    Check the YOLO models training course by forwarding it in real time.
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
                res = forward_yolo(net, img)
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


def train():
    """
    train the YOLO network using the hyper parameter at the top.
    """
    global lr, batch_size, epoch, test_img_path, class_names, class_count, validation_ratio, new_model_saved

    total_image_paths = glob(f'{train_image_path}/*lane*etc*/*.jpg')
    random.shuffle(total_image_paths)
    train_image_count = int(len(total_image_paths) * (1 - validation_ratio))
    train_image_paths = total_image_paths[:train_image_count]
    validation_image_paths = total_image_paths[train_image_count:]
    train_data_generator = YoloDataGenerator(image_paths=train_image_paths, augmentation=False)
    validation_data_generator = YoloDataGenerator(image_paths=validation_image_paths, augmentation=False)

    print(f'train image count : {len(train_image_paths)}')
    print(f'validation image count : {len(validation_image_paths)}')

    model_input = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], img_channels))

    x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, kernel_initializer='he_uniform', padding='same')(model_input)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    sc = x
    sc = tf.keras.layers.MaxPool2D()(sc)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, kernel_initializer='he_uniform', padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=5, kernel_initializer='he_uniform', padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=5, kernel_initializer='he_uniform', padding='same')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Concatenate()([x, sc])
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=class_count + 5, kernel_size=1, activation='sigmoid')(x)
    model = tf.keras.models.Model(model_input, x)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr, rho=0.9), loss=MeanAbsoluteLogError())
    model.save('model.h5')

    live_view(total_image_paths)
    model.fit(
        x=train_data_generator,
        validation_data=validation_data_generator,
        epochs=epoch,
        callbacks=[
            # tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/sgd_fpw_loss_epoch_{epoch}_loss_{loss:.4f}_val_loss_{val_loss:.4f}.h5'),
            tf.keras.callbacks.ModelCheckpoint(filepath='model.h5'),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=new_model_saved_on),
        ]
    )
    model.save('model.h5')

    freeze('model.h5')
    net = cv2.dnn.readNet('model.pb')
    for cur_img_path in glob(f'{test_img_path}/*/*.jpg'):
        print(cur_img_path)
        img = cv2.imread(cur_img_path, cv2.IMREAD_COLOR)
        img = resize(img, (input_shape[1], input_shape[0]))
        st = time()
        res = forward_yolo(net, img)
        et = time()
        print(f'[Inference Time] : {(et - st):.3f} s')
        img = bounding_box(img, res)
        print(res)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    net = cv2.dnn.readNet('model.pb')
    print(f'\nconvert pb success. {len(net.getLayerNames())} layers')


def test_video():
    global input_shape, img_channels
    class_names.append('license_plate')
    net = cv2.dnn.readNet('model.pb')

    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (640, 368))

    # cap = cv2.VideoCapture('rtsp://admin:samsungg2b!@samsungg2bcctv.iptime.org:1500/video1s1')
    cap = cv2.VideoCapture(r'C:\inz\videos\truen.mkv')
    # cap = cv2.VideoCapture(r'C:\inz\videos\hc_4k_18_day.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon_not_trained.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (2).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (3).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (4).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (5).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\noon (6).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\night.mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\night (2).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\night (3).mp4')
    # cap = cv2.VideoCapture(r'C:\inz\videos\night (4).mp4')

    while True:
        frame_exist, x = cap.read()
        if not frame_exist:
            break
        # x = resize(x, (input_shape[1], input_shape[0]))
        x = cv2.resize(x, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        res = forward_yolo(net, x)
        boxed = bounding_box(x, res)
        # out.write(x)
        cv2.imshow('res', x)
        if ord('q') == cv2.waitKey(1):
            break
    # out.release()
    cap.release()
    cv2.destroyAllWindows()


def count_test():
    global img_type
    YoloDataGenerator.init_label()
    freeze('over_fit_model.h5')
    net = cv2.dnn.readNet('model.pb')
    freeze('lp_type_model_1202.h5')
    ltc_net = cv2.dnn.readNet('model.pb')
    total_image_paths = glob(f'{train_image_path}/*/*.jpg')
    classes = [0 for _ in range(8)]

    from tqdm import tqdm
    for path in tqdm(total_image_paths):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = resize(x, (input_shape[1], input_shape[0]))
        boxes = forward_yolo(net, x)
        for box in boxes:
            x1, y1, x2, y2 = box['box']
            plate = x[y1:y2, x1:x2]
            plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            plate = resize(plate, (192, 96))
            plate_x = np.asarray(plate).reshape(1, 1, 96, 192).astype('float32') / 255.
            ltc_net.setInput(plate_x)
            res = ltc_net.forward_yolo()
            res = np.asarray(res).reshape(8, )
            max_index = int(np.argmax(res))
            if res[max_index] > 0.8:
                classes[max_index] += 1
    print(classes)
    # x = bounding_box(x, boxes)
    # cv2.imshow('x', x)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # count_test()
    train()
    # freeze('model.h5')
    # test_video()
