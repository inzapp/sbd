from glob import glob

import cv2
import numpy as np

import sbd


def test_label():
    """
    test sbd label(yolo format) is validate
    """
    with open(f'{sbd.train_image_path}/classes.txt', 'rt') as classes_file:
        sbd.class_names = [s.replace('\n', '') for s in classes_file.readlines()]
    img_paths = glob(rf'{sbd.train_image_path}\*.jpg')
    for cur_img_path in img_paths:
        x = cv2.imread(cur_img_path, sbd.img_type)
        x = sbd.resize(x, (sbd.input_shape[1], sbd.input_shape[0]))
        converted_res = []
        with open(rf'{cur_img_path[:-4]}.txt', mode='rt') as f:
            lines = f.readlines()
        y = np.zeros((sbd.input_shape[0], sbd.input_shape[1]), dtype=np.uint8)
        for line in lines:
            class_index, cx, cy, w, h = list(map(float, line.split(' ')))
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1, y1, x2, y2 = int(x1 * x.shape[1]), int(y1 * x.shape[0]), int(x2 * x.shape[1]), int(y2 * x.shape[0])
            cv2.rectangle(y, (x1, y1), (x2, y2), (255, 255, 255), -1)
            converted_res.append({'class': class_index, 'box': [x1, y1, x2, y2]})
        y = sbd.SbdDataGenerator.compress(y)
        y = (y * 255.).astype('uint8')
        y = cv2.resize(y, (sbd.input_shape[1], sbd.input_shape[0]), interpolation=cv2.INTER_NEAREST)
        x = sbd.bounding_box(x, converted_res)
        cv2.imshow('x', x)
        cv2.imshow('y', y)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_label()
