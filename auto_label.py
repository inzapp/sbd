import os
import cv2
import numpy as np
import shutil as sh
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from yolo import Yolo
from util import ModelUtil
from concurrent.futures.thread import ThreadPoolExecutor


g_confidence_threshold = 0.5
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def auto_label(model_path, image_path, origin_classes_txt_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    _, _, channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
    color_mode = cv2.IMREAD_GRAYSCALE if channel == 1 else cv2.IMREAD_COLOR

    image_paths = glob(f'{image_path}/*.jpg')
    classes_txt_path = f'{image_path}/classes.txt'
    try:
        sh.copy(origin_classes_txt_path, classes_txt_path)
    except sh.SameFileError:
        pass

    pool = ThreadPoolExecutor(8)
    fs = []
    for path in image_paths:
        fs.append(pool.submit(ModelUtil.load_img, path, channel))

    # device = ModelUtil.available_device()
    device = 'cpu'
    for f in tqdm(fs):
        raw, _, path = f.result()
        boxes = Yolo.predict(model, raw, device, confidence_threshold=g_confidence_threshold)
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
    origin_classes_txt_path = r'/train_data/imagenet/train/classes.txt'
    img_path = r'/train_data/unlabeled/imagenet'
    auto_label(model_path, img_path, origin_classes_txt_path)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()

