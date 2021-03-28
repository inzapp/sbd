import os
from glob import glob

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

iou_thresholds = [0.5]
confidence_thresholds = np.asarray(list(range(5, 100, 5))).astype('float32') / 100.0


def calc_precision_recall(y, label_lines, iou_threshold, confidence_threshold, class_index):
    precision = 1.0
    recall = 1.0
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
