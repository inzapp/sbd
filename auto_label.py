"""
Authors : inzapp

Github url : https://github.com/inzapp/sbd

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
import numpy as np
import shutil as sh
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from sbd import SBD
from util import Util
from concurrent.futures.thread import ThreadPoolExecutor


g_confidence_threshold = 0.5
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def auto_label(model_path, image_path, origin_classes_txt_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    channel = input_shape[-1]

    image_paths = glob(f'{image_path}/*.jpg')
    classes_txt_path = f'{image_path}/classes.txt'
    try:
        sh.copy(origin_classes_txt_path, classes_txt_path)
    except sh.SameFileError:
        pass

    pool = ThreadPoolExecutor(8)
    fs = []
    for path in image_paths:
        fs.append(pool.submit(Util.load_img, path, channel))

    for f in tqdm(fs):
        raw, _, path = f.result()
        boxes = SBD.predict(model, raw, 'cpu', confidence_threshold=g_confidence_threshold)
        label_content = ''
        for box in boxes:
            class_index = box['class']
            xmin, ymin, xmax, ymax = box['bbox_norm']
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + (w * 0.5)
            cy = ymin + (h * 0.5)
            cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
            label_content += f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'
        with open(f'{path[:-4]}.txt', 'wt') as file:
            file.write(label_content)


def main():
    model_path = r'./checkpoint/model/m0/best.h5'
    img_path = r'/train_data/unlabeled/imagenet'
    origin_classes_txt_path = r'/train_data/imagenet/train/classes.txt'
    auto_label(model_path, img_path, origin_classes_txt_path)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()

