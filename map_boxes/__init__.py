"""
Author: Roman Solovyev, IPPM RAS
URL: https://github.com/ZFTurbo

Code based on: https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/eval.py
"""

import numpy as np
import pandas as pd
try:
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=False)
    from .compute_overlap import compute_overlap
except:
    print("Couldn't import fast version of function compute_overlap, will use slow one. Check cython intallation")
    from .compute_overlap_slow import compute_overlap


def get_real_annotations(table):
    res = dict()
    ids = table['ImageID'].values.astype(np.str)
    labels = table['LabelName'].values.astype(np.str)
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
    ids = table['ImageID'].values.astype(np.str)
    labels = table['LabelName'].values.astype(np.str)
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


def mean_average_precision_for_boxes(ann, pred, iou_threshold=0.5, confidence_threshold_for_f1=0.25, exclude_not_in_annotations=False, verbose=True):
    """
    :param ann: path to CSV-file with annotations or numpy array of shape (N, 6)
    :param pred: path to CSV-file with predictions (detections) or numpy array of shape (N, 7)
    :param iou_threshold: IoU between boxes which count as 'match'. Default: 0.5
    :param exclude_not_in_annotations: exclude image IDs which are not exist in annotations. Default: False
    :param verbose: print detailed run info. Default: True
    :return: tuple, where first value is mAP and second values is dict with AP for each class.
    """

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
        print('Number of files in annotations: {}'.format(len(ann_unique)))
        print('Number of files in predictions: {}'.format(len(preds_unique)))

    # Exclude files not in annotations!
    if exclude_not_in_annotations:
        preds = preds[preds['ImageID'].isin(ann_unique)]
        preds_unique = preds['ImageID'].unique()
        if verbose:
            print('Number of files in detection after reduction: {}'.format(len(preds_unique)))

    unique_classes = valid['LabelName'].unique().astype(np.str)
    if verbose:
        print('Unique classes: {}'.format(len(unique_classes)))

    all_detections = get_detections(preds)
    all_annotations = get_real_annotations(valid)
    if verbose:
        print('Detections length: {}'.format(len(all_detections)))
        print('Annotations length: {}'.format(len(all_annotations)))

    print(f'\nconfidence threshold for tp, fp, fn calculate : {confidence_threshold_for_f1}')
    total_tp_iou_sum = 0.0
    total_tp = 0
    total_fp = 0
    total_obj_count = 0
    average_precisions = {}
    class_confidence_sum = 0.0
    for class_index, label in enumerate(sorted(unique_classes)):
        # Negative class
        if str(label) == 'nan':
            continue

        tp_ious = []
        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0
        for i in range(len(ann_unique)):
            detections = []
            annotations = []
            id = ann_unique[i]
            if id in all_detections:
                if label in all_detections[id]:
                    detections = all_detections[id][label]
            if id in all_annotations:
                if label in all_annotations[id]:
                    annotations = all_annotations[id][label]

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
                    continue

                overlaps = compute_overlap(np.expand_dims(np.array(d, dtype=np.float64), axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                    tp_ious.append(max_overlap[0])
                    # print(f'conf : {d[4]:.4f}, iou : {max_overlap[0]:.4f}')
                else:
                    false_positives.append(1)
                    true_positives.append(0)
                    tp_ious.append(0.0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)
        tp_ious = np.array(tp_ious)

        tp_confidence = np.sum(scores * true_positives) / np.sum(true_positives)
        class_confidence_sum += tp_confidence

        # mask
        tp_mask= np.where(scores > confidence_threshold_for_f1, 1, 0)
        true_positives_over_threshold = true_positives * tp_mask
        false_positives_over_threshold = false_positives * tp_mask
        tp_ious *= tp_mask
        tp_iou_sum = np.sum(tp_ious)

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
        average_precisions[label] = average_precision, num_annotations
        if verbose:
            print(f'class {class_index} ap : {average_precision:.4f}, obj count : {obj_count:6d}, tp : {tp:6d}, fp : {fp:6d}, fn : {fn:6d}, precision : {p:.4f}, recall : {r:.4f}, f1 : {f1:.4f}, iou : {tp_iou:.4f}, confidence : {tp_confidence:.4f}')

    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
    mean_ap = precision / present_classes
    p = total_tp / (total_tp + total_fp + 1e-7)
    r = total_tp / (total_obj_count + 1e-7)
    f1 = (2.0 * p * r) / (p + r + 1e-7)
    tp_iou = total_tp_iou_sum / (total_tp + 1e-7)
    confidence = class_confidence_sum / present_classes
    print(f'F1@{int(iou_threshold * 100)} : {f1:.4f}')
    print(f'mAP@{int(iou_threshold * 100)} : {mean_ap:.4f}')
    print(f'TP_IOU@{int(iou_threshold * 100)} : {tp_iou:.4f}')
    print(f'TP_Confidence : {confidence:.4f}')
    return mean_ap, f1, tp_iou, total_tp, total_fp, total_obj_count - total_tp
