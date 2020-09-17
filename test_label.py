from glob import glob

import cv2

import sbd

import numpy as np


def test_label():
    with open(rf'{sbd.train_img_path}\classes.txt', 'rt') as classes_file:
        sbd.class_names = classes_file.readlines()
    img_paths = glob(rf'{sbd.train_img_path}\*.jpg') + glob(rf'{sbd.train_img_path}\*.png')
    for cur_img_path in img_paths:
        img = cv2.imread(cur_img_path, sbd.img_type)
        img_height, img_width, c = img.shape
        file_name_without_extension = cur_img_path.replace('\\', '/').split('/').pop().split('.')[0]
        converted_res = []
        with open(rf'{sbd.train_img_path}/{file_name_without_extension}.txt', 'rt') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            class_index, cx, cy, w, h = np.array(line.split(' ')).astype('float32')
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1, y1, x2, y2 = int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height)
            converted_res.append({'class': class_index, 'box': [x1, y1, x2, y2]})
        img = sbd.bounding_box(img, converted_res)
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_label()
