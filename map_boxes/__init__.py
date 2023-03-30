"""
Author: Roman Solovyev, IPPM RAS
URL: https://github.com/ZFTurbo

Code based on: https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/eval.py
"""
import os
import numpy as np
import pandas as pd
# try:
#     import pyximport
#     pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=False)
#     from .compute_overlap import compute_overlap
# except:
#     print("Couldn't import fast version of function compute_overlap, will use slow one. Check cython intallation")
from .compute_overlap_slow import compute_overlap


def get_real_annotations(table):
    res = dict()
    ids = list(map(str, table['ImageID'].values))
    labels = list(map(str, table['LabelName'].values))
    xmin = table['XMin'].values.astype(np.float32)
    xmax = table['XMax'].values.astype(np.float32)
    ymin = table['YMin'].values.astype(np.float32)
    ymax = table['YMax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i]]
        res[id][label].append(box)

    return res


def get_detections(table):
    res = dict()
    ids = list(map(str, table['ImageID'].values))
    labels = list(map(str, table['LabelName'].values))
    scores = table['Conf'].values.astype(np.float32)
    xmin = table['XMin'].values.astype(np.float32)
    xmax = table['XMax'].values.astype(np.float32)
    ymin = table['YMin'].values.astype(np.float32)
    ymax = table['YMax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i], scores[i]]
        res[id][label].append(box)

    return res


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _print(msg, txt_content):
    print(msg)
    txt_content += f'{msg}\n'
    return txt_content


def mean_average_precision_for_boxes(ann, pred, iou_threshold=0.5, confidence_threshold_for_f1=0.25, exclude_not_in_annotations=False, verbose=True, classes_txt_path=''):
    """
    :param ann: path to CSV-file with annotations or numpy array of shape (N, 6)
    :param pred: path to CSV-file with predictions (detections) or numpy array of shape (N, 7)
    :param iou_threshold: IoU between boxes which count as 'match'. Default: 0.5
    :param exclude_not_in_annotations: exclude image IDs which are not exist in annotations. Default: False
    :param verbose: print detailed run info. Default: True
    :param classes_txt_path: class names file for show result. Default: ''
    :return: tuple, where first value is mAP and second values is dict with AP for each class.
    """

    class_names = []
    max_class_name_len = 1
    if classes_txt_path != '':
        if os.path.exists(classes_txt_path) and os.path.isfile(classes_txt_path):
            with open(classes_txt_path, 'rt') as f:
                lines = f.readlines()
            for line in lines:
                class_name = line.replace('\n', '')
                if len(class_name) > max_class_name_len:
                    max_class_name_len = len(class_name)
                class_names.append(class_name)

    if isinstance(ann, str):
        valid = pd.read_csv(ann)
    else:
        valid = pd.DataFrame(ann, columns=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'])

    if isinstance(pred, str):
        preds = pd.read_csv(pred)
    else:
        preds = pd.DataFrame(pred, columns=['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax'])

    ann_unique = valid['ImageID'].unique()
    preds_unique = preds['ImageID'].unique()

    print()
    txt_content = ''
    unique_classes = list(map(str, valid['LabelName'].unique()))
    if verbose:
        txt_content = _print(f'Unique classes: {len(unique_classes)}', txt_content)

    if verbose:
        txt_content = _print(f'Number of files in annotations: {len(ann_unique)}', txt_content)
        txt_content = _print(f'Number of files in predictions: {len(preds_unique)}', txt_content)

    # Exclude files not in annotations!
    if exclude_not_in_annotations:
        preds = preds[preds['ImageID'].isin(ann_unique)]
        preds_unique = preds['ImageID'].unique()
        if verbose:
            txt_content = _print(f'Number of files in detection after reduction: {len(preds_unique)}', txt_content)

    all_detections = get_detections(preds)
    all_annotations = get_real_annotations(valid)
    # if verbose:
    #     txt_content = _print('Detections length: {}'.format(len(all_detections)), txt_content)
    #     txt_content = _print('Annotations length: {}'.format(len(all_annotations)), txt_content)

    txt_content = _print(f'\nconfidence threshold for tp, fp, fn calculate : {confidence_threshold_for_f1}', txt_content)
    total_tp_iou_sum = 0.0
    total_tp_confidence_sum = 0.0
    total_tp = 0
    total_fp = 0
    total_obj_count = 0
    average_precisions = {}
    for _, class_index_str in enumerate(sorted(unique_classes, key=lambda x: int(x))):
        # Negative class
        if str(class_index_str) == 'nan':
            continue

        tp_ious = []
        tp_confidences = []
        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0
        for i in range(len(ann_unique)):
            detections = []
            annotations = []
            id = ann_unique[i]
            if id in all_detections:
                if class_index_str in all_detections[id]:
                    detections = all_detections[id][class_index_str]
            if id in all_annotations:
                if class_index_str in all_annotations[id]:
                    annotations = all_annotations[id][class_index_str]

            if len(detections) == 0 and len(annotations) == 0:
                continue

            num_annotations += len(annotations)
            detected_annotations = []

            annotations = np.array(annotations, dtype=np.float64)
            for d in detections:
                scores.append(d[4])

                if len(annotations) == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    tp_ious.append(0.0)
                    tp_confidences.append(0.0)
                    continue

                overlaps = compute_overlap(np.expand_dims(np.array(d, dtype=np.float64), axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                    tp_ious.append(max_overlap[0])
                    tp_confidences.append(d[4])
                    # print(f'conf : {d[4]:.4f}, iou : {max_overlap[0]:.4f}')
                else:
                    false_positives.append(1)
                    true_positives.append(0)
                    tp_ious.append(0.0)
                    tp_confidences.append(0.0)

        if num_annotations == 0:
            average_precisions[class_index_str] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)
        tp_ious = np.array(tp_ious)
        tp_confidences = np.array(tp_confidences)

        # mask
        tp_mask = np.where(scores > confidence_threshold_for_f1, 1, 0)
        true_positives_over_threshold = true_positives * tp_mask
        false_positives_over_threshold = false_positives * tp_mask
        tp_ious *= tp_mask
        tp_iou_sum = np.sum(tp_ious)
        tp_confidences *= tp_mask
        tp_confidence_sum = np.sum(tp_confidences)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        obj_count = int(num_annotations)
        tp = int(np.sum(true_positives_over_threshold))
        fp = int(np.sum(false_positives_over_threshold))
        fn = obj_count - tp
        p = tp / (tp + fp + 1e-7)
        r = tp / (obj_count + 1e-7)
        f1 = (2.0 * p * r) / (p + r + 1e-7)
        tp_iou = tp_iou_sum / (tp + 1e-7)
        total_tp_iou_sum += tp_iou_sum
        tp_confidence = tp_confidence_sum / (tp + 1e-7)
        total_tp_confidence_sum += tp_confidence_sum

        total_obj_count += obj_count
        total_tp += tp
        total_fp += fp

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[class_index_str] = average_precision, num_annotations

        if verbose:
            class_index = int(class_index_str)
            class_name = f'class {class_index_str}'
            if len(class_names) >= class_index + 1:
                class_name = class_names[class_index]
            txt_content = _print(f'{class_name:{max_class_name_len}s} ap : {average_precision:.4f}, obj count : {obj_count:6d}, tp : {tp:6d}, fp : {fp:6d}, fn : {fn:6d}, precision : {p:.4f}, recall : {r:.4f}, f1 : {f1:.4f}, iou : {tp_iou:.4f}, confidence : {tp_confidence:.4f}', txt_content)

    present_classes = 0
    precision = 0
    for _, (average_precision, num_annotations) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
    mean_ap = precision / (present_classes + 1e-7)
    p = total_tp / (total_tp + total_fp + 1e-7)
    r = total_tp / (total_obj_count + 1e-7)
    f1 = (2.0 * p * r) / (p + r + 1e-7)
    tp_iou = total_tp_iou_sum / (total_tp + 1e-7)
    tp_confidence = total_tp_confidence_sum / (total_tp + 1e-7)
    txt_content = _print(f'F1@{int(iou_threshold * 100)} : {f1:.4f}', txt_content)
    txt_content = _print(f'mAP@{int(iou_threshold * 100)} : {mean_ap:.4f}', txt_content)
    txt_content = _print(f'TP_IOU@{int(iou_threshold * 100)} : {tp_iou:.4f}', txt_content)
    txt_content = _print(f'TP_Confidence : {tp_confidence:.4f}', txt_content)
    return mean_ap, f1, tp_iou, total_tp, total_fp, total_obj_count - total_tp, tp_confidence, txt_content

