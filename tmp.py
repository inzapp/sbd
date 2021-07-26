import os
from glob import glob

import cv2
import numpy as np

# [[100m], [200m]]
roi_of = {
    '0': [[0.29, 0.24, 0.62, 0.48], [0.27, 0.18, 0.43, 0.27]],
    '1': [[0.05, 0.45, 0.49, 0.76], [0.39, 0.30, 0.67, 0.40]],
    '2': [[0.25, 0.35, 0.98, 0.92], [0.35, 0.03, 0.86, 0.40]],
    '3': [[0.05, 0.31, 0.68, 0.85], [0.52, 0.09, 0.77, 0.32]],
    '4': [[0.01, 0.35, 0.78, 0.87], [0.03, 0.00, 0.66, 0.52]],
    '5': [[0.08, 0.32, 0.73, 0.84], [0.36, 0.10, 0.78, 0.44]],
    '6': [[0.10, 0.35, 0.74, 0.89], [0.10, 0.35, 0.74, 0.89]],
    '7': [[0.21, 0.39, 0.94, 0.84], [0.17, 0.21, 0.65, 0.47]],
    '8': [[0.33, 0.59, 0.77, 0.95], [0.37, 0.44, 0.70, 0.68]],
    '9': [[0.21, 0.40, 0.93, 0.83], [0.12, 0.13, 0.63, 0.45]],
    '10': [[0.29, 0.40, 0.59, 0.60], [0.29, 0.33, 0.46, 0.45]],
    '11': [[0.24, 0.37, 0.76, 0.83], [0.30, 0.20, 0.58, 0.38]],
    '12': [[0.26, 0.55, 0.77, 0.91], [0.37, 0.47, 0.66, 0.66]],
    '13': [[0.24, 0.36, 0.89, 0.83], [0.21, 0.17, 0.62, 0.37]],
    '14': [[0.02, 0.40, 0.80, 0.91], [0.01, 0.14, 0.44, 0.40]],
    '15': [[0.10, 0.43, 0.80, 0.96], [0.31, 0.20, 0.67, 0.47]],
    '16': [[0.05, 0.36, 0.85, 0.95], [0.02, 0.19, 0.36, 0.49]],
    '17': [[0.01, 0.34, 0.79, 0.93], [0.39, 0.10, 0.77, 0.42]],
    '18': [[0.13, 0.44, 0.79, 0.91], [0.22, 0.07, 0.79, 0.41]],
    '19': [[0.01, 0.33, 0.85, 0.91], [0.02, 0.06, 0.61, 0.33]],
    '20': [[0.07, 0.09, 0.46, 0.36], [0.05, 0.03, 0.83, 0.25]],
    '21': [[0.36, 0.31, 0.99, 0.87], [0.62, 0.04, 0.96, 0.32]],
    '22': [[0.02, 0.33, 0.52, 0.77], [0.04, 0.01, 0.58, 0.33]],
    '23': [[0.01, 0.03, 0.47, 0.32], [0.45, 0.01, 0.82, 0.31]],
    '24': [[0.03, 0.37, 0.92, 0.96], [0.00, 0.01, 0.62, 0.45]],
    '25': [[0.02, 0.39, 0.72, 0.90], [0.02, 0.16, 0.67, 0.53]],
    '26': [[0.13, 0.29, 0.77, 0.90], [0.16, 0.06, 0.76, 0.43]],
    '27': [[0.01, 0.32, 0.63, 0.86], [0.01, 0.03, 0.53, 0.27]],
    '28': [[0.01, 0.33, 0.69, 0.82], [0.25, 0.32, 0.85, 0.71]],
    '29': [[0.01, 0.10, 0.89, 0.86], [0.05, 0.01, 0.98, 0.35]],
    '30': [[0.04, 0.27, 0.89, 0.94], [0.01, 0.09, 0.47, 0.40]],
    '31': [[0.07, 0.30, 0.78, 0.81], [0.25, 0.16, 0.70, 0.44]],
    '32': [[0.26, 0.38, 0.80, 0.79], [0.18, 0.27, 0.65, 0.58]],
    '33': [[0.01, 0.33, 0.72, 0.95], [0.21, 0.19, 0.62, 0.51]],
    '34': [[0.29, 0.47, 0.87, 0.88], [0.17, 0.35, 0.58, 0.61]],
    '35': [[0.02, 0.66, 0.46, 0.97], [0.00, 0.43, 0.25, 0.67]],
    '36': [[0.48, 0.28, 0.96, 0.56], [0.27, 0.05, 0.65, 0.43]],
    '37': [[0.33, 0.44, 0.95, 0.84], [0.17, 0.18, 0.63, 0.45]],
    '38': [[0.11, 0.22, 0.97, 0.94], [0.26, 0.01, 0.87, 0.37]],
    '39': [[0.17, 0.44, 0.93, 0.94], [0.39, 0.26, 0.75, 0.55]],
    '40': [[0.02, 0.40, 0.97, 0.84], [0.12, 0.14, 0.71, 0.46]],
    '41': [[0.09, 0.23, 0.96, 0.87], [0.28, 0.04, 0.91, 0.43]],
    '42': [[0.00, 0.56, 0.52, 0.93], [0.00, 0.35, 0.03, 0.66]],
    '43': [[0.01, 0.27, 0.98, 0.90], [0.15, 0.02, 0.84, 0.42]],
    '44': [[0.01, 0.27, 0.98, 0.90], [0.15, 0.02, 0.84, 0.42]],
    '45': [[0.01, 0.27, 0.98, 0.90], [0.15, 0.02, 0.84, 0.42]],
    '46': [[0.03, 0.37, 0.95, 0.96], [0.16, 0.04, 0.85, 0.42]],
    '47': [[0.05, 0.33, 0.97, 0.92], [0.37, 0.10, 0.84, 0.47]],
    '48': [[0.05, 0.33, 0.97, 0.92], [0.37, 0.10, 0.84, 0.47]],
    '49': [[0.13, 0.32, 0.91, 0.72], [0.22, 0.21, 0.87, 0.52]],
    '50': [[0.10, 0.35, 0.93, 0.84], [0.18, 0.11, 0.89, 0.51]],
    '51': [[0.10, 0.35, 0.93, 0.84], [0.18, 0.11, 0.89, 0.51]],
    '52': [[0.10, 0.35, 0.93, 0.84], [0.18, 0.11, 0.89, 0.51]],
    '53': [[0.10, 0.35, 0.93, 0.84], [0.18, 0.11, 0.89, 0.51]],
    '54': [[0.20, 0.38, 0.92, 0.91], [0.34, 0.13, 0.95, 0.46]],
}


def roi_crop_with_label_convert(path, roi):
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1
    # label_path = f'{path[:-4]}.txt'
    # if not os.path.exists(label_path):
    #     print(f'label not exist : {label_path}')
    #     return

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    raw_height, raw_width = img.shape[0], img.shape[1]

    roi_x1_s32 = int(roi_x1 * raw_width)
    roi_x2_s32 = int(roi_x2 * raw_width)
    roi_y1_s32 = int(roi_y1 * raw_height)
    roi_y2_s32 = int(roi_y2 * raw_height)

    img = img[roi_y1_s32:roi_y2_s32, roi_x1_s32:roi_x2_s32]
    cv2.imshow('img', img)
    cv2.waitKey(0)
    return

    with open(label_path, 'rt') as f:
        lines = f.readlines()

    new_label_content = ''
    for line in lines:
        class_index, cx, cy, w, h = list(map(float, line.replace('\n', '').split()))
        class_index = int(class_index)

        cx = (cx - roi_x1) / roi_w
        cy = (cy - roi_y1) / roi_h
        w = w / roi_w
        h = h / roi_h

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        x1, y1, x2, y2 = np.clip(np.array([x1, y1, x2, y2]), 0.0, 1.0)

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        new_label_content += f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'
    # cv2.imwrite(path, img)
    # with open(label_path, 'wt') as f:
    #     f.writelines(new_label_content)


def main():
    for cur_dir_path in glob(r'C:\inz\tmp\roi_crop\*'):
        dir_name = cur_dir_path.replace('\\', '/').split('/')[-1]
        for img_path in glob(rf'{cur_dir_path}/*.jpg'):
            for cur_roi in roi_of[dir_name]:
                roi_crop_with_label_convert(img_path, cur_roi)


if __name__ == '__main__':
    main()
