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


def get_y_true(label_lines, target_class_index):
    raw_width = 1000
    raw_height = 1000
    y_true = []
    for label_line in label_lines:
        class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
        if int(class_index) != target_class_index:
            continue
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

            class_index = -1
            max_percentage = -1
            for cur_channel_index in range(5, channels):
                if max_percentage < y[i][j][cur_channel_index]:
                    class_index = cur_channel_index - 5
                    max_percentage = y[i][j][cur_channel_index]
            if class_index != target_class_index:
                continue

            cx_f = j / float(cols) + 1.0 / float(cols) * y[i][j][1]
            cy_f = i / float(rows) + 1.0 / float(rows) * y[i][j][2]
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

            y_pred.append({
                'confidence': confidence,
                'bbox': [x_min, y_min, x_max, y_max],
                'class': class_index,
                'discard': False})
    # return y_pred

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
    y_true = get_y_true(label_lines, target_class_index)
    num_obj = len(y_true)
    if num_obj == 0:
        return None, None, None, None, None, None

    y_pred = get_y_pred(y, confidence_threshold, target_class_index)

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
    precision = tp / (float(len(y_pred)) + 1e-5)

    """
    recall = True Positive / True Positive + False Negative
    recall = True Positive / All Ground Truths
    """
    recall = tp / (float(len(y_true)) + 1e-5)
    return precision, recall, num_obj, tp, fp, fn


def calc_ap(precisions, recalls):
    for i in range(len(recalls)):
        for j in range(len(recalls)):
            if i == j:
                continue
            if recalls[i] < recalls[j]:
                tmp = recalls[i]
                recalls[i] = recalls[j]
                recalls[j] = tmp

                tmp = precisions[i]
                precisions[i] = precisions[j]
                precisions[j] = tmp

    recall_check = np.asarray(list(range(100))).astype('float32') / 100.0
    head_recall = list()
    head_precision = list()
    for i in range(len(recall_check)):
        if recalls[0] > recall_check[i]:
            head_precision.append(1.0)
            head_recall.append(recall_check[i])
        else:
            break

    precisions = head_precision + list(precisions)
    recalls = head_recall + list(recalls)

    # interpolate precisions
    sorted_pure_precisions = sorted(list(set(precisions)), reverse=True)
    indexed_pure_precisions = list()
    prev_max_index = -1
    for i in range(len(sorted_pure_precisions)):
        max_index = -1
        for j in range(len(precisions)):
            if sorted_pure_precisions[i] == precisions[j]:
                max_index = j
        if max_index > prev_max_index:
            indexed_pure_precisions.append({'max_index': max_index, 'val': sorted_pure_precisions[i]})
            prev_max_index = max_index
    if len(indexed_pure_precisions) > 1:
        for i in range(1, len(indexed_pure_precisions)):
            start_index = indexed_pure_precisions[i - 1]['max_index'] + 1
            end_index = indexed_pure_precisions[i]['max_index']
            for interpolation_index in range(start_index, end_index + 1):
                precisions[interpolation_index] = indexed_pure_precisions[i]['val']

    if recalls[-1] < 1.0:
        precisions[-1] = 0.0

    ap = 0.0
    for i in range(len(precisions) - 1):
        ap += precisions[i] * (recalls[i + 1] - recalls[i])

    # print(ap)
    # plt.figure()
    # plt.step(recalls, precisions)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.1])
    # plt.xlim([0.0, 1.1])
    # plt.show()

    return ap


@tf.function
def predict_on_graph(__model, __x):
    return __model(__x, training=False)


def calc_mean_average_precision(model_path, image_paths, class_names_file_path=''):
    global iou_thresholds, confidence_thresholds
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    input_size = (input_shape[1], input_shape[0])
    color_mode = cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR
    num_classes = model.output_shape[-1] - 5

    obj_count = np.zeros((len(iou_thresholds), num_classes, len(confidence_thresholds)), dtype=np.int32)
    valid_count = np.zeros((len(iou_thresholds), num_classes, len(confidence_thresholds)), dtype=np.int32)
    precisions = np.zeros((len(iou_thresholds), num_classes, len(confidence_thresholds)), dtype=np.float32)
    recalls = np.zeros((len(iou_thresholds), num_classes, len(confidence_thresholds)), dtype=np.float32)
    tps = np.zeros((len(iou_thresholds), num_classes, len(confidence_thresholds)), dtype=np.int32)
    fps = np.zeros((len(iou_thresholds), num_classes, len(confidence_thresholds)), dtype=np.int32)
    fns = np.zeros((len(iou_thresholds), num_classes, len(confidence_thresholds)), dtype=np.int32)

    for image_path in tqdm(image_paths):
        label_path = f'{image_path[:-4]}.txt'
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            continue
        with open(label_path, mode='rt') as f:
            label_lines = f.readlines()
        if len(label_lines) == 0:
            continue
        x = cv2.imread(image_path, color_mode)
        x = cv2.resize(x, input_size)
        x = np.asarray(x).astype('float32').reshape((1,) + input_shape) / 255.0
        y = np.asarray(predict_on_graph(model, x))[0]  # 3D array

        for iou_index, iou_threshold in enumerate(iou_thresholds):
            for class_index in range(num_classes):
                for confidence_index, confidence_threshold in enumerate(confidence_thresholds):
                    precision, recall, num_obj, tp, fp, fn = calc_precision_recall(y, label_lines, iou_threshold, confidence_threshold, class_index)
                    if precision is None and recall is None:
                        continue
                    valid_count[iou_index][class_index][confidence_index] += 1
                    precisions[iou_index][class_index][confidence_index] += precision
                    recalls[iou_index][class_index][confidence_index] += recall

                    obj_count[iou_index][class_index][confidence_index] += num_obj
                    tps[iou_index][class_index][confidence_index] += tp
                    fps[iou_index][class_index][confidence_index] += fp
                    fns[iou_index][class_index][confidence_index] += fn

    for iou_index in range(len(iou_thresholds)):
        for class_index in range(num_classes):
            for confidence_index in range(len(confidence_thresholds)):
                if valid_count[iou_index][class_index][confidence_index] == 0:
                    continue
                precisions[iou_index][class_index][confidence_index] /= valid_count[iou_index][class_index][confidence_index]
                recalls[iou_index][class_index][confidence_index] /= valid_count[iou_index][class_index][confidence_index]

    mean_ap_sum = 0.0
    for iou_index, iou_threshold in enumerate(iou_thresholds):
        class_ap_sum = 0.0
        for class_index in range(num_classes):
            cur_class_precisions = precisions[iou_index][class_index]
            cur_class_recalls = recalls[iou_index][class_index]
            cur_class_ap = calc_ap(cur_class_precisions, cur_class_recalls)
            class_ap_sum += cur_class_ap

            cur_class_obj_count = obj_count[iou_index][class_index][0]
            cur_class_tp = tps[iou_index][class_index][0]
            cur_class_fp = fps[iou_index][class_index][0]
            cur_class_fn = fns[iou_index][class_index][0]
            print(f'class {str(class_index):3s} ap : {cur_class_ap:.4f}, obj_count : {str(cur_class_obj_count):5s}, tp : {str(cur_class_tp):5s}, fp : {str(cur_class_fp):5s}, fn : {str(cur_class_fn):5s}')
        mean_ap = class_ap_sum / float(num_classes)
        mean_ap_sum += mean_ap
        print(f'mAP@{int(iou_threshold * 100)} : {mean_ap:.4f}\n')
    return mean_ap_sum / len(iou_thresholds)


if __name__ == '__main__':
    calc_mean_average_precision(
        r'C:\inz\git\yolo-lab\checkpoints\person_info_detector_epoch_36_loss_1.8349_val_loss_14.9586.h5',
        glob(r'\\192.168.101.200\train_data\person_data_validation\*.jpg'),
        class_names_file_path=r'\\192.168.101.200\train_data\person_data\classes.txt')

    # calc_mean_average_precision(
    #     r'C:\inz\git\yolo-lab\checkpoints\lcd_288_144\lcd_b1_epoch_250_loss_0.3989_val_loss_6.0434.h5',
    #     glob(r'C:\inz\train_data\lp_character_detection\lcd_b1\*\*.jpg'),
    #     class_names_file_path=r'C:\inz\train_data\lp_character_detection\lcd_b1\classes.txt')
