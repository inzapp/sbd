import os
from glob import glob

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

iou_thresholds = [0.5]
confidence_thresholds = np.asarray(list(range(5, 100, 5))).astype('float32') / 100.0
nms_iou_threshold = 0.5


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


def get_y_true(label_lines):
    raw_width = 1000
    raw_height = 1000
    y_true = []
    for label_line in label_lines:
        class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
        x1 = int((cx - w / 2.0) * raw_width)
        x2 = int((cx + w / 2.0) * raw_width)
        y1 = int((cy - h / 2.0) * raw_height)
        y2 = int((cy + h / 2.0) * raw_height)
        y_true.append({
            'class': int(class_index),
            'bbox': [x1, y1, x2, y2],
            'discard': False})
    return y_true


def get_y_pred(y, confidence_threshold, target_class_index):
    global nms_iou_threshold
    raw_width = 1000
    raw_height = 1000
    rows, cols, channels = y.shape[0], y.shape[1], y.shape[2]

    y_pred = []
    for i in range(rows):
        for j in range(cols):
            confidence = y[i][j][0]
            if confidence < confidence_threshold:
                continue
            cx_f = j / float(cols) + 1 / float(cols) * y[i][j][1]
            cy_f = i / float(rows) + 1 / float(rows) * y[i][j][2]
            w = y[i][j][3]
            h = y[i][j][4]

            x_min_f = cx_f - w / 2.0
            y_min_f = cy_f - h / 2.0
            x_max_f = cx_f + w / 2.0
            y_max_f = cy_f + h / 2.0
            x_min = int(x_min_f * raw_width)
            y_min = int(y_min_f * raw_height)
            x_max = int(x_max_f * raw_width)
            y_max = int(y_max_f * raw_height)
            class_index = -1
            max_percentage = -1
            for cur_channel_index in range(5, channels):
                if max_percentage < y[i][j][cur_channel_index]:
                    class_index = cur_channel_index
                    max_percentage = y[i][j][cur_channel_index]
            class_index -= 5
            if class_index != target_class_index:
                continue
            y_pred.append({
                'confidence': confidence,
                'bbox': [x_min, y_min, x_max, y_max],
                'class': class_index,
                'discard': False})

    for i in range(len(y_pred)):
        if y_pred[i]['discard']:
            continue
        for j in range(len(y_pred)):
            if i == j or y_pred[j]['discard']:
                continue
            if iou(y_pred[i]['bbox'], y_pred[j]['bbox']) > nms_iou_threshold:
                if y_pred[i]['confidence'] >= y_pred[j]['confidence']:
                    y_pred[j]['discard'] = True

    y_pred_copy = np.asarray(y_pred.copy())
    y_pred = []
    for i in range(len(y_pred_copy)):
        if not y_pred_copy[i]['discard']:
            y_pred.append(y_pred_copy[i])
    return y_pred


def calc_precision_recall(y, label_lines, iou_threshold, confidence_threshold, target_class_index):
    y_true = get_y_true(label_lines)
    y_pred = get_y_pred(y, confidence_threshold, target_class_index)

    # print(y_true)
    # print(y_pred)
    # print()

    tp = 0
    for i in range(len(y_true)):
        for j in range(len(y_pred)):
            if y_pred[j]['discard'] or y_true[i]['class'] != y_pred[j]['class']:
                continue
            if iou(y_true[i]['bbox'], y_pred[j]['bbox']) > iou_threshold:
                y_true[i]['discard'] = True
                y_pred[j]['discard'] = True
                tp += 1
                break

    fp = 0
    for i in range(len(y_pred)):
        if not y_pred[i]['discard']:
            fp += 1

    fn = 0
    for i in range(len(y_true)):
        if not y_true[i]['discard']:
            fn += 1

    """
    precision = True Positive / True Positive + False Positive
    precision = True Positive / All Detections
    """
    precision = tp / float(tp + fp + 1e-5)

    """
    recall = True Positive / True Positive + False Negative
    recall = True Positive / All Ground Truths
    """
    recall = tp / float(tp + fn + 1e-5)

    # print(precision)
    # print(recall)
    # print()
    return precision, recall


def calc_ap(precisions, recalls):
    return 1.0


@tf.function
def predict_on_graph(__model, __x):
    return __model(__x, training=False)


def main(model_path, image_paths, class_names_file_path=''):
    global iou_thresholds, confidence_thresholds
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    input_size = (input_shape[1], input_shape[0])
    color_mode = cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR
    num_classes = model.output_shape[-1] - 5

    valid_image_count = 0  # mAP 계산에 사용된 이미지 개수
    class_ap_sum = [0.0 for _ in range(num_classes)]  # 클래스별 mAP를 구하기 위한 누적 변수 = sum of mean ap per iou
    iou_threshold_mean_ap_sum = [0.0 for _ in range(len(iou_thresholds))]  # iou 별 mAP를 구하기 위한 누적 변수

    for image_path in tqdm(image_paths):
        label_path = f'{image_path[:-4]}.txt'
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            continue
        with open(label_path, mode='rt') as f:
            label_lines = f.readlines()
        if len(label_lines) == 0:
            continue
        valid_image_count += 1
        x = cv2.imread(image_path, color_mode)
        x = cv2.resize(x, input_size)
        x = np.asarray(x).astype('float32').reshape((1,) + input_shape) / 255.0
        y = np.asarray(predict_on_graph(model, x))[0]  # 3D array

        for iou_threshold in iou_thresholds:
            cur_image_class_ap_sum = 0.0  # 현재 이미지의 mAP를 구하기 위한 누적변수 = sum of image class ap
            for class_index in range(num_classes):
                precisions = list()
                recalls = list()
                for confidence_threshold in confidence_thresholds:
                    precision, recall = calc_precision_recall(y, label_lines, iou_threshold, confidence_threshold, class_index)
                    precisions.append(precision)
                    recalls.append(recall)
                cur_image_class_ap = calc_ap(precisions, recalls)
                cur_image_class_ap_sum += cur_image_class_ap
                class_ap_sum[class_index] += cur_image_class_ap  # 반복 끝나고 이걸 이미지 개수로 나눠야 한다
            cur_image_mean_ap = cur_image_class_ap_sum / float(num_classes)
            iou_threshold_mean_ap_sum[iou_thresholds.index(iou_threshold)] += cur_image_mean_ap

    for i in range(num_classes):
        class_ap_sum[i] = class_ap_sum[i] / float(valid_image_count)
        print(f'class {i} ap : {class_ap_sum[i]:.4f}')

    for i in range(len(iou_thresholds)):
        iou_threshold_mean_ap_sum[i] = iou_threshold_mean_ap_sum[i] / float(valid_image_count)
        print(f'mAP@{int(iou_thresholds[i] * 100)} : {iou_threshold_mean_ap_sum[i]:.4f}\n')


if __name__ == '__main__':
    main(
        'loon_detector_model_epoch_184_f1_0.9651_val_f1_0.8532.h5',
        glob('C:/inz/train_data/loon_detection_train/*.jpg'),
        class_names_file_path=r'C:\inz\train_data\loon_detection_train\classes.txt')
