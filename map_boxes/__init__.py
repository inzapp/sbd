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


def _print(msg, txt_content, verbose):
    if verbose:
        print(msg)
    txt_content += f'{msg}\n'
    return txt_content


def calculate_f1_score(num_annotations, true_positives, false_positives, scores, tp_ious, tp_confidences, confidence_threshold):
    false_positives_copy = np.array(false_positives)
    true_positives_copy = np.array(true_positives)
    scores_copy = np.array(scores)
    tp_ious_copy = np.array(tp_ious)
    tp_confidences_copy = np.array(tp_confidences)

    # mask
    tp_mask = np.where(scores_copy > confidence_threshold, 1, 0)
    true_positives_over_threshold = true_positives_copy * tp_mask
    false_positives_over_threshold = false_positives_copy * tp_mask
    tp_ious_copy *= tp_mask
    tp_iou_sum = np.sum(tp_ious_copy)
    tp_confidences_copy *= tp_mask
    tp_confidence_sum = np.sum(tp_confidences_copy)

    # sort by score
    indices = np.argsort(-scores_copy)
    false_positives_copy = false_positives_copy[indices]
    true_positives_copy = true_positives_copy[indices]

    obj_count = int(num_annotations)
    tp = int(np.sum(true_positives_over_threshold))
    fp = int(np.sum(false_positives_over_threshold))
    fn = obj_count - tp
    eps = 1e-7
    p = tp / (tp + fp + eps)
    r = tp / (obj_count + eps)
    f1 = (2.0 * p * r) / (p + r + eps)
    tp_iou = tp_iou_sum / (tp + eps)
    tp_confidence = tp_confidence_sum / (tp + eps)

    ret = {}
    ret['confidence_threshold'] = confidence_threshold
    ret['true_positives'] = true_positives_copy
    ret['false_positives'] = false_positives_copy
    ret['obj_count'] = obj_count
    ret['tp_iou'] = tp_iou
    ret['tp_iou_sum'] = tp_iou_sum
    ret['tp_confidence'] = tp_confidence
    ret['tp_confidence_sum'] = tp_confidence_sum
    ret['tp'] = tp
    ret['fp'] = fp
    ret['fn'] = fn
    ret['f1'] = f1
    ret['p'] = p
    ret['r'] = r
    return ret


def mean_average_precision_for_boxes(ann, pred, iou_threshold=0.5, confidence_threshold_for_f1=0.25, exclude_not_in_annotations=False, verbose=True, find_best_threshold=False, classes_txt_path=''):
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
    else:
        max_class_name_len = 9

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

    if verbose:
        print()
    txt_content = ''
    unique_classes = list(map(str, valid['LabelName'].unique()))
    txt_content = _print(f'Unique classes: {len(unique_classes)}', txt_content, verbose)

    txt_content = _print(f'Number of files in annotations: {len(ann_unique)}', txt_content, verbose)
    txt_content = _print(f'Number of files in predictions: {len(preds_unique)}', txt_content, verbose)

    # Exclude files not in annotations!
    if exclude_not_in_annotations:
        preds = preds[preds['ImageID'].isin(ann_unique)]
        preds_unique = preds['ImageID'].unique()
        txt_content = _print(f'Number of files in detection after reduction: {len(preds_unique)}', txt_content, verbose)

    all_detections = get_detections(preds)
    all_annotations = get_real_annotations(valid)

    txt_content = _print(f'\nNMS iou threshold : {iou_threshold}', txt_content, verbose)
    if find_best_threshold:
        txt_content = _print(f'confidence threshold for tp, fp, fn calculate : best confidence policy per class', txt_content, verbose)
    else:
        txt_content = _print(f'confidence threshold for tp, fp, fn calculate : {confidence_threshold_for_f1}', txt_content, verbose)
    total_tp_iou_sum = 0.0
    total_tp_confidence_sum = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
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

        ap_ret = calculate_f1_score(num_annotations, true_positives, false_positives, scores, tp_ious, tp_confidences, confidence_threshold_for_f1)
        best_ret = ap_ret
        if find_best_threshold:
            best_f1 = 0.0
            patience_count = 0
            for i in range(99):
                class_confidence_threshold = i / 100.0
                cur_ret = calculate_f1_score(num_annotations, true_positives, false_positives, scores, tp_ious, tp_confidences, class_confidence_threshold)
                cur_f1 = cur_ret['f1']
                if cur_f1 > best_f1:
                    best_f1 = cur_f1
                    best_ret = cur_ret
                else:
                    patience_count += 1
                    if patience_count == 5:
                        break

        true_positives = ap_ret['true_positives']  # use ap_ret
        false_positives = ap_ret['false_positives']  # use ap_ret

        confidence_threshold = best_ret['confidence_threshold']
        obj_count = best_ret['obj_count']
        tp_iou = best_ret['tp_iou']
        tp_iou_sum = best_ret['tp_iou_sum']
        tp_confidence = best_ret['tp_confidence']
        tp_confidence_sum = best_ret['tp_confidence_sum']
        tp = best_ret['tp']
        fp = best_ret['fp']
        fn = best_ret['fn']
        f1 = best_ret['f1']
        p = best_ret['p']
        r = best_ret['r']

        total_obj_count += obj_count
        total_tp_iou_sum += tp_iou_sum
        total_tp_confidence_sum += tp_confidence_sum
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[class_index_str] = average_precision, num_annotations

        class_index = int(class_index_str)
        class_name = f'class {class_index_str}'
        if len(class_names) >= class_index + 1:
            class_name = class_names[class_index]
        txt_content = _print(f'{class_name:{max_class_name_len}s} AP: {average_precision:.4f}, Labels: {obj_count:6d}, TP: {tp:6d}, FP: {fp:6d}, FN: {fn:6d}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}, IoU: {tp_iou:.4f}, Confidence: {tp_confidence:.4f}, Threshold: {confidence_threshold:.2f}', txt_content, verbose)

    present_classes = 0
    precision = 0
    for _, (average_precision, num_annotations) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision

    eps = 1e-7
    mean_ap = precision / (present_classes + eps)
    p = total_tp / (total_tp + total_fp + eps)
    r = total_tp / (total_obj_count + eps)
    f1 = (2.0 * p * r) / (p + r + eps)
    tp_iou = total_tp_iou_sum / (total_tp + eps)
    tp_confidence = total_tp_confidence_sum / (total_tp + eps)
    class_name = 'total'
    txt_content = _print(f'\n{class_name:{max_class_name_len}s} AP: {mean_ap:.4f}, Labels: {total_obj_count:6d}, TP: {total_tp:6d}, FP: {total_fp:6d}, FN: {total_fn:6d}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}, IoU: {tp_iou:.4f}, Confidence: {tp_confidence:.4f}', txt_content, verbose)
    return mean_ap, f1, tp_iou, total_tp, total_fp, total_obj_count - total_tp, tp_confidence, txt_content

