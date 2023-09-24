import os
import numpy as np

from tqdm import tqdm
from map_boxes import mean_average_precision_for_boxes
from concurrent.futures.thread import ThreadPoolExecutor


g_iou_threshold = 0.5
g_confidence_threshold = 0.2  # only for tp, fp, fn
g_nms_iou_threshold = 0.45  # darknet yolo nms threshold value
g_annotations_csv_name = 'annotations.csv'
g_predictions_csv_name = 'predictions.csv'


def load_label(image_path, unknown_class_index):
    csv = ''
    label_path = f'{image_path[:-4]}.txt'
    basename = os.path.basename(image_path)
    with open(label_path, 'rt') as f:
        lines = f.readlines()
    for line in lines:
        class_index, cx, cy, w, h = list(map(float, line.split()))
        class_index = int(class_index)
        if class_index == unknown_class_index:
            continue
        xmin = cx - w * 0.5
        ymin = cy - h * 0.5
        xmax = cx + w * 0.5
        ymax = cy + h * 0.5
        xmin, ymin, xmax, ymax = np.clip(np.array([xmin, ymin, xmax, ymax]), 0.0, 1.0)
        csv += f'{basename},{class_index},{xmin:.6f},{xmax:.6f},{ymin:.6f},{ymax:.6f}\n'
    return csv


def make_annotations_csv(image_paths, unknown_class_index, csv_path):
    fs = []
    pool = ThreadPoolExecutor(8)
    for path in image_paths:
        fs.append(pool.submit(load_label, path, unknown_class_index))
    csv = 'ImageID,LabelName,XMin,XMax,YMin,YMax\n'
    for f in tqdm(fs, desc='annotations csv creation'):
        csv += f.result()
    with open(csv_path, 'wt') as f:
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


def make_predictions_csv(model, image_paths, device, csv_path):
    from sbd import SBD
    from util import Util
    fs = []
    input_channel = model.input_shape[1:][-1]
    pool = ThreadPoolExecutor(8)
    for path in image_paths:
        fs.append(pool.submit(Util.load_img, path, input_channel))
    csv = 'ImageID,LabelName,Conf,XMin,XMax,YMin,YMax\n'
    for f in tqdm(fs, desc='predictions csv creation'):
        img, _, path = f.result()
        boxes = SBD.predict(model, img, confidence_threshold=0.001, nms_iou_threshold=g_nms_iou_threshold, device=device)
        csv += convert_boxes_to_csv_lines(path, boxes)
    with open(csv_path, 'wt') as f:
        f.writelines(csv)


def calc_mean_average_precision(
    model,
    image_paths,
    device,
    unknown_class_index,
    confidence_threshold,
    tp_iou_threshold,
    classes_txt_path,
    annotations_csv_path,
    predictions_csv_path,
    cached,
    find_best_threshold):
    global g_iou_threshold, g_confidence_threshold, g_annotations_csv_name, g_predictions_csv_name
    if not cached:
        make_annotations_csv(image_paths, unknown_class_index, annotations_csv_path)
        make_predictions_csv(model, image_paths, device, predictions_csv_path)
    if find_best_threshold:
        best_f1 = 0.0
        best_ret = None
        best_txt_content = None
        best_confidence_threshold = 0.0
        patience = 10
        patience_count = 0
        print('start find best confidence threshold for f1')
        for i in range(1, 100, 1):
            confidence_threshold = i / 100.0
            ret = mean_average_precision_for_boxes(
                ann=annotations_csv_path,
                pred=predictions_csv_path,
                confidence_threshold_for_f1=confidence_threshold,
                iou_threshold=tp_iou_threshold,
                classes_txt_path=classes_txt_path,
                verbose=False)
            mean_ap, f1, tp_iou, total_tp, total_fp, total_fn, tp_confidence, txt_content = ret
            print(f'confidence threshold({confidence_threshold:.2f}) : {f1:.4f}')
            if f1 > best_f1:
                best_f1 = f1
                best_ret = ret
                best_txt_content = txt_content
                best_confidence_threshold = confidence_threshold
                patience_count = 0
            else:
                patience_count += 1
                if patience_count == patience:
                    break
        print(f'\nbest confidence threshold for f1 : {best_confidence_threshold:.2f}\n')
        print(best_txt_content)
        return best_ret
    else:
        return mean_average_precision_for_boxes(
            ann=annotations_csv_path,
            pred=predictions_csv_path,
            confidence_threshold_for_f1=confidence_threshold,
            iou_threshold=tp_iou_threshold,
            classes_txt_path=classes_txt_path,
            verbose=True)


if __name__ == '__main__':
    mean_average_precision_for_boxes(g_annotations_csv_name, g_predictions_csv_name, confidence_threshold_for_f1=g_confidence_threshold, iou_threshold=g_iou_threshold, verbose=True)

