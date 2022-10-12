import os
import shutil as sh
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tqdm import tqdm

from yolo import Yolo
from util import ModelUtil


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
g_confidence_threshold = 0.5


def auto_label(model_path, image_path, origin_classes_txt_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    w, h, channel = ModelUtil.get_width_height_channel_from_input_shape(input_shape)
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

    device = 'cpu'
    for d in tf.config.list_physical_devices():
        if str(d).find('GPU') > -1:
            device = 'gpu'
            break

    for f in tqdm(fs):
        raw, _, path = f.result()
        res = Yolo.predict(model, raw, device, confidence_threshold=g_confidence_threshold)
        label_content = ''
        for i in range(len(res)):
            class_index = res[i]['class']
            xmin, ymin, xmax, ymax = res[i]['bbox_norm']
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w * 0.5
            cy = ymin + h * 0.5
            label_content += f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'

        with open(f'{path[:-4]}.txt', 'wt') as file:
            file.writelines(label_content)


def main():
    model_path = r'C:\inz\git\yolo-lab\checkpoints\model_3000_iter_mAP_1.0000_f1_1.0000_iou_0.8663_tp_104_fp_0_fn_0_conf_0.8196_ul_all.h5'
    origin_classes_txt_path = r'C:\inz\train_data\loon\validation_copy\classes.txt'
    img_path = r'C:\inz\train_data\loon\validation_copy'
    auto_label(model_path, img_path, origin_classes_txt_path)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()

