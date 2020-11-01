from glob import glob

import cv2

import sbd


def test_label():
    """
    test if sbd label(yolo format) is validate
    """
    with open(rf'{sbd.train_img_path}\classes.txt', 'rt') as classes_file:
        sbd.class_names = classes_file.readlines()
    img_paths = glob(rf'{sbd.train_img_path}\*.jpg') + glob(rf'{sbd.train_img_path}\*.png')
    for cur_img_path in img_paths:
        img = cv2.imread(cur_img_path, sbd.img_type)
        converted_res = []
        with open(rf'{cur_img_path[:-4]}.txt', mode='rt') as f:
            lines = f.readlines()
        for line in lines:
            class_index, cx, cy, w, h = list(map(float, line.split(' ')))
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
            converted_res.append({'class': class_index, 'box': [x1, y1, x2, y2]})
        img = sbd.bounding_box(img, converted_res)
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_label()
