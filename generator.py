from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np
import tensorflow as tf


class YoloDataGenerator:
    def __init__(self, train_image_path, input_shape, output_shape, batch_size, validation_split=0.0):
        self.class_names, self.num_classes = self.init_class_names(train_image_path)
        train_image_paths = self.init_image_paths(train_image_path)
        train_image_paths, validation_image_paths = self.split_paths(train_image_paths, validation_split)
        self.train_generator_flow = GeneratorFlow(train_image_paths, input_shape, output_shape, batch_size, 'training')
        self.validation_generator_flow = GeneratorFlow(validation_image_paths, input_shape, output_shape, batch_size, 'validation')

    @staticmethod
    def init_image_paths(train_image_path):
        image_paths = []
        image_paths += glob(f'{train_image_path}/*/*.jpg')
        image_paths += glob(f'{train_image_path}/*/*.png')
        image_paths = np.asarray(image_paths)
        for i in range(len(image_paths)):
            image_paths[i] = image_paths[i].replace('\\', '/')
        return sorted(image_paths)

    @staticmethod
    def init_class_names(train_image_path):
        """
        Init YOLO label from classes.txt file.
        """
        with open(f'{train_image_path}/classes.txt', 'rt') as classes_file:
            class_names = [s.replace('\n', '') for s in classes_file.readlines()]
            num_classes = len(class_names)
        return class_names, num_classes

    @staticmethod
    def split_paths(train_image_paths, validation_split):
        assert 0.0 <= validation_split <= 1.0
        train_image_paths = np.asarray(train_image_paths)
        if validation_split == 0.0:
            return train_image_paths, np.asarray([])
        r = np.arange(len(train_image_paths))
        np.random.shuffle(r)
        train_image_paths = train_image_paths[r]
        num_train_image_paths = int(len(train_image_paths) * (1.0 - validation_split))
        train_image_paths = train_image_paths[:num_train_image_paths]
        validation_image_paths = train_image_paths[num_train_image_paths:]
        return train_image_paths, validation_image_paths

    def flow(self, subset='training'):
        if subset == 'training':
            return self.train_generator_flow
        elif subset == 'validation':
            return self.validation_generator_flow


class GeneratorFlow(tf.keras.utils.Sequence):
    """
    Custom data generator for YOLO model.
    Usage:
        generator = GeneratorFlow(image_paths=train_image_paths)
    """

    def __init__(self, image_paths, input_shape, output_shape, batch_size, subset='training'):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.subset = subset
        self.random_indexes = np.arange(len(self.image_paths))
        self.pool = ThreadPoolExecutor(8)
        np.random.shuffle(self.random_indexes)

    def load_img(self, path):
        return path, cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        start_index = index * self.batch_size
        fs = []
        for i in range(start_index, start_index + self.batch_size):
            fs.append(self.pool.submit(self.load_img, self.image_paths[self.random_indexes[i]]))
        for f in fs:
            cur_img_path, x = f.result()
            if x.shape[1] > self.input_shape[1] or x.shape[0] > self.input_shape[0]:
                x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_AREA)
            else:
                x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_LINEAR)
            x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
            batch_x.append(x)

            with open(f'{cur_img_path[:-4]}.txt', mode='rt') as file:
                label_lines = file.readlines()
            y = [np.zeros(self.output_shape, dtype=np.float32) for _ in range(self.output_shape[2])]
            grid_width_ratio = 1 / float(self.output_shape[1])
            grid_height_ratio = 1 / float(self.output_shape[0])
            for label_line in label_lines:
                class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
                center_row = int(cy * self.output_shape[0])
                center_col = int(cx * self.output_shape[1])
                y[0][center_row][center_col] = 1.0
                y[1][center_row][center_col] = (cx - (center_col * grid_width_ratio)) / grid_width_ratio
                y[2][center_row][center_col] = (cy - (center_row * grid_height_ratio)) / grid_height_ratio
                y[3][center_row][center_col] = w
                y[4][center_row][center_col] = h
                y[int(class_index + 5)][center_row][center_col] = 1.0
            y = np.moveaxis(np.asarray(y), 0, -1).reshape(self.output_shape)
            batch_y.append(y)
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.random_indexes)
