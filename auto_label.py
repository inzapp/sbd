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


def iou(a, b):
    """
    Intersection of union function.
    :param a: [x_min, y_min, x_max, y_max] format box a
    :param b: [x_min, y_min, x_max, y_max] format box b
    """
    a_x_min, a_y_min, a_x_max, a_y_max = a
    b_x_min, b_y_min, b_x_max, b_y_max = b
    intersection_width = min(a_x_max, b_x_max) - max(a_x_min, b_x_min)
    intersection_height = min(a_y_max, b_y_max) - max(a_y_min, b_y_min)
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
    b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
    union_area = a_area + b_area - intersection_area
    return intersection_area / (float(union_area) + 1e-5)


def get_y_pred(y):
    global g_nms_iou_threshold, g_confidence_threshold
    raw_width = 1000
    raw_height = 1000

    y_pred = []
    for layer_index in range(len(y)):
        rows = y[layer_index][0].shape[0]
        cols = y[layer_index][0].shape[1]
        channels = y[layer_index][0].shape[2]

        for i in range(rows):
            for j in range(cols):
                confidence = y[layer_index][0][i][j][0]
                if confidence < g_confidence_threshold:  # darknet yolo mAP confidence threshold value
                    continue

                class_index = -1
                class_score = 0.0
                for cur_channel_index in range(5, channels):
                    cur_class_score = y[layer_index][0][i][j][cur_channel_index]
                    if class_score < cur_class_score:
                        class_index = cur_channel_index - 5
                        class_score = cur_class_score

                cx_f = j / float(cols) + 1.0 / float(cols) * y[layer_index][0][i][j][1]
                cy_f = i / float(rows) + 1.0 / float(rows) * y[layer_index][0][i][j][2]
                w = y[layer_index][0][i][j][3]
                h = y[layer_index][0][i][j][4]

                x_min_f = cx_f - w / 2.0
                y_min_f = cy_f - h / 2.0
                x_max_f = cx_f + w / 2.0
                y_max_f = cy_f + h / 2.0
                x_min = int(x_min_f * raw_width)
                y_min = int(y_min_f * raw_height)
                x_max = int(x_max_f * raw_width)
                y_max = int(y_max_f * raw_height)

                y_pred.append({
                    'confidence': confidence,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'bbox_norm': [cx_f, cy_f, w, h],
                    'class': class_index,
                    'result': '',
                    'precision': 0.0,
                    'recall': 0.0,
                    'discard': False})

    for i in range(len(y_pred)):
        if y_pred[i]['discard']:
            continue
        for j in range(len(y_pred)):
            if i == j or y_pred[j]['discard']:
                continue
            if iou(y_pred[i]['bbox'], y_pred[j]['bbox']) > g_nms_iou_threshold:
                if y_pred[i]['confidence'] >= y_pred[j]['confidence']:
                    y_pred[j]['discard'] = True

    y_pred_copy = np.asarray(y_pred.copy())
    y_pred = []
    for i in range(len(y_pred_copy)):
        if not y_pred_copy[i]['discard']:
            y_pred.append(y_pred_copy[i])
    return y_pred


def load_x_image_path(image_path, color_mode, input_size, input_shape):
    x = cv2.imread(image_path, color_mode)
    x = cv2.resize(x, input_size)
    x = np.asarray(x).astype('float32').reshape((1,) + input_shape) / 255.0
    return x, image_path


def auto_label(model_path, image_path, origin_classes_txt_path):
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
    for image_path in image_paths:
        fs.append(pool.submit(load_x_image_path, image_path, color_mode, input_size, input_shape))

    for f in tqdm(fs):
        x, image_path = f.result()
        y = model.predict_on_batch(x=x)
        y_pred = get_y_pred(y)

        label_content = ''
        for i in range(len(y_pred)):
            class_index = y_pred[i]['class']
            # if class_index != 0:
            #     continue
            cx, cy, w, h = y_pred[i]['bbox_norm']
            label_content += f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'

        with open(f'{image_path[:-4]}.txt', 'wt') as file:
            file.writelines(label_content)


def main():
    pass
    # model_path = r'C:\inz\git\yolo-lab\checkpoints\auto_label_200m\model_400000_iter_mAP_0.2873_f1_0.7198_tp_iou_0.7606.h5'
    # origin_classes_txt_path = r'J:\200m_sequence\2_1664\0_100\classes.txt'
    img_path = r'J:\200M_2021_1012_ADD\87_yj1012_5_sbd_ing'
    # for dir_path in glob(fr'J:\200m_sequence\2_1664\*'):
    #     if not os.path.isdir(dir_path):
    #         continue
    #     img_path = dir_path
    #     auto_label(model_path, img_path, origin_classes_txt_path)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
