import tensorflow as tf


def precision(y_true, y_pred):
    tp = tf.reduce_sum(y_pred[:, :, :, 0] * y_true[:, :, :, 0])
    return tp / (tf.reduce_sum(y_pred[:, :, :, 0]) + 1e-5)


def recall(y_true, y_pred):
    tp = tf.reduce_sum(y_pred[:, :, :, 0] * y_true[:, :, :, 0])
    return tp / (tf.reduce_sum(y_true[:, :, :, 0]) + 1e-5)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (p * r * 2.0) / (p + r + 1e-5)


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
    if intersection_width < 0.0 or intersection_height < 0.0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
    b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
    union_area = a_area + b_area - intersection_area
    return intersection_area / float(union_area)


def iou_f1(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    bbox_true = []
    for batch_index in range(len(y_true)):
        for i in range(len(y_true[batch_index])):
            for j in range(len(y_true[batch_index][i])):
                if y_true[batch_index][i][j][0] == 1.0:
                    cx = y_true[batch_index][i][j][1]
                    cy = y_true[batch_index][i][j][2]
                    w = y_true[batch_index][i][j][3]
                    h = y_true[batch_index][i][j][4]
                    x1, y1 = cx - w / 2.0, cy - h / 2.0
                    x2, y2 = cx + w / 2.0, cy + h / 2.0
                    x1 = int(x1 * 1000.0)
                    y1 = int(y1 * 1000.0)
                    x2 = int(x2 * 1000.0)
                    y2 = int(y2 * 1000.0)
                    class_index = -1
                    max_value = 0.0
                    for channel_index in range(5, len(y_true[batch_index][i][j])):
                        if y_true[batch_index][i][j][channel_index] > max_value:
                            max_value = y_true[batch_index][i][j][channel_index]
                            class_index = channel_index - 5
                    bbox_true.append([x1, y1, x2, y2, class_index])

    confidence_threshold = 0.25
    bbox_pred = []
    for batch_index in range(len(y_pred)):
        for i in range(len(y_pred[batch_index])):
            for j in range(len(y_pred[batch_index][i])):
                if y_pred[batch_index][i][j][0] >= confidence_threshold:
                    cx = y_pred[batch_index][i][j][1]
                    cy = y_pred[batch_index][i][j][2]
                    w = y_pred[batch_index][i][j][3]
                    h = y_pred[batch_index][i][j][4]
                    x1, y1 = cx - w / 2.0, cy - h / 2.0
                    x2, y2 = cx + w / 2.0, cy + h / 2.0
                    x1 = int(x1 * 1000.0)
                    y1 = int(y1 * 1000.0)
                    x2 = int(x2 * 1000.0)
                    y2 = int(y2 * 1000.0)
                    class_index = -1
                    max_value = 0.0
                    for channel_index in range(5, len(y_pred[batch_index][i][j])):
                        if y_pred[batch_index][i][j][channel_index] > max_value:
                            max_value = y_pred[batch_index][i][j][channel_index]
                            class_index = channel_index - 5
                    bbox_pred.append([x1, y1, x2, y2, class_index])

    tp, fp, fn = 0, 0, 0
    for cur_bbox_true in bbox_true:
        for cur_bbox_pred in bbox_pred:
            
            pass
