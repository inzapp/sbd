import os
import cv2
import numpy as np
from tqdm import tqdm
from map_boxes import mean_average_precision_for_boxes
from concurrent.futures.thread import ThreadPoolExecutor


g_iou_threshold = 0.5
g_confidence_threshold = 0.25  # only for tp, fp, fn
g_nms_iou_threshold = 0.45  # darknet yolo nms threshold value
g_annotations_csv_name = 'annotations.csv'
g_predictions_csv_name = 'predictions.csv'


def load_label(image_path):
    csv = ''
    label_path = f'{image_path[:-4]}.txt'
    basename = os.path.basename(image_path)
    with open(label_path, 'rt') as f:
        lines = f.readlines()
    for line in lines:
        class_index, cx, cy, w, h = list(map(float, line.split()))
        class_index = int(class_index)
        xmin = cx - w * 0.5
        ymin = cy - h * 0.5
        xmax = cx + w * 0.5
        ymax = cy + h * 0.5
        csv += f'{basename},{class_index},{xmin:.6f},{xmax:.6f},{ymin:.6f},{ymax:.6f}\n'
    return csv


def make_annotations_csv(image_paths):
    global g_annotations_csv_name
    print('annotations csv creation start')
    fs = []
    pool = ThreadPoolExecutor(8)
    for path in image_paths:
        fs.append(pool.submit(load_label, path))
    csv = 'ImageID,LabelName,XMin,XMax,YMin,YMax\n'
    for f in tqdm(fs):
        csv += f.result()
    with open(g_annotations_csv_name, 'wt') as f:
        f.writelines(csv)


def convert_boxes_to_csv_lines(path, boxes):
    csv = ''
    for b in boxes:
        basename = os.path.basename(path)
        confidence = b['confidence']
        class_index = b['class']
        xmin, ymin, xmax, ymax = b['bbox_norm']
        csv += f'{basename},{class_index},{confidence:.6f},{xmin:.6f},{xmax:.6f},{ymin:.6f},{ymax:.6f}\n'
    return csv


def make_predictions_csv(model, image_paths):
    from yolo import Yolo
    print('predictions csv creation start')
    global g_predictions_csv_name
    csv = 'ImageID,LabelName,Conf,XMin,XMax,YMin,YMax\n'
    for path in tqdm(image_paths):
        img = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if model.input_shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = Yolo.predict(model, img, confidence_threshold=0.005, nms_iou_threshold=g_nms_iou_threshold)
        csv += convert_boxes_to_csv_lines(path, boxes)
    with open(g_predictions_csv_name, 'wt') as f:
        f.writelines(csv)


def calc_mean_average_precision(model, all_image_paths):
    global g_iou_threshold, g_confidence_threshold, g_annotations_csv_name, g_predictions_csv_name
    # from random import shuffle; shuffle(all_image_paths); image_paths = all_image_paths[:500] # test
    image_paths = all_image_paths
    make_annotations_csv(image_paths)
    make_predictions_csv(model, image_paths)
    return mean_average_precision_for_boxes(g_annotations_csv_name, g_predictions_csv_name, confidence_threshold_for_f1=g_confidence_threshold, iou_threshold=g_iou_threshold, verbose=True)


if __name__ == '__main__':
    mean_average_precision_for_boxes(g_annotations_csv_name, g_predictions_csv_name, confidence_threshold_for_f1=g_confidence_threshold, iou_threshold=g_iou_threshold, verbose=True)
