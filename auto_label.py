import os
import shutil as sh
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

g_confidence_threshold = 0.9
g_nms_iou_threshold = 0.45


def load_img_with_path(image_path, color_mode, input_size, input_shape):
    img = cv2.imdecode(np.fromfile(image_path, np.uint8), color_mode)
    return img, image_path


def auto_label(model_path, image_path, origin_classes_txt_path):
    from yolo import Yolo
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    input_size = (input_shape[1], input_shape[0])
    color_mode = cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR

    image_paths = glob(f'{image_path}/*.jpg')
    classes_txt_path = f'{image_path}/classes.txt'
    try:
        sh.copy(origin_classes_txt_path, classes_txt_path)
    except sh.SameFileError:
        pass

    pool = ThreadPoolExecutor(8)
    fs = []
    for path in image_paths:
        fs.append(pool.submit(load_img_with_path, path, color_mode, input_size, input_shape))

    for f in tqdm(fs):
        img, path = f.result()
        res = Yolo.predict(model, img)
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
    model_path = r'C:\inz\git\yolo-lab\checkpoints\model_7000_iter_mAP_1.0000_f1_1.0000_tp_iou_0.9181_tp_52_fp_0_fn_0_ul_all.h5'
    origin_classes_txt_path = r'C:\inz\tmp\square_set\classes.txt'
    img_path = r'C:\inz\tmp\square_set'
    auto_label(model_path, img_path, origin_classes_txt_path)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
