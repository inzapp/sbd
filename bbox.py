"""
Authors : inzapp

Github : https://github.com/inzapp/sbd
"""
import numpy as np


class BoundingBox:
    def __init__(self, class_index, cx, cy, w, h, confidence=0.0, clip=True):
        self.confidence = confidence
        self.class_index = int(class_index)
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.x1 = self.cx - (self.w * 0.5)
        self.y1 = self.cy - (self.h * 0.5)
        self.x2 = self.cx + (self.w * 0.5)
        self.y2 = self.cy + (self.h * 0.5)
        if clip:
            self.x1, self.y1, self.x2, self.y2 = np.clip(np.array([self.x1, self.y1, self.x2, self.y2]), 0.0, 1.0)
            self.w = self.x2 - self.x1
            self.h = self.y2 - self.y1
            self.cx = self.x1 + (self.w * 0.5)
            self.cy = self.y1 + (self.h * 0.5)

    def get_cxcywh(self):
        return self.cx, self.cy, self.w, self.h

    def get_x1y1x2y2(self):
        return self.x1, self.y1, self.x2, self.y2

    @staticmethod
    def convert_cxcywh_to_x1y1x2y2(cx, cy, w, h):
        x1 = cx - (w * 0.5)
        y1 = cy - (h * 0.5)
        x2 = cx + (w * 0.5)
        y2 = cy + (h * 0.5)
        return x1, y1, x2, y2

    @staticmethod
    def convert_x1y1x2y2_to_cxcywh(x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        cx = x1 + (w * 0.5)
        cy = y1 + (h * 0.5)
        return cx, cy, w, h

    @staticmethod
    def iou(box_a, box_b):
        a_x1, a_y1, a_x2, a_y2 = box_a.get_x1y1x2y2()
        b_x1, b_y1, b_x2, b_y2 = box_b.get_x1y1x2y2()
        intersection_width = min(a_x2, b_x2) - max(a_x1, b_x1)
        intersection_height = min(a_y2, b_y2) - max(a_y1, b_y1)
        if intersection_width <= 0 or intersection_height <= 0:
            return 0.0
        intersection_area = intersection_width * intersection_height
        a_area = abs((a_x2 - a_x1) * (a_y2 - a_y1))
        b_area = abs((b_x2 - b_x1) * (b_y2 - b_y1))
        union_area = a_area + b_area - intersection_area
        return intersection_area / (float(union_area) + 1e-5)

    @staticmethod
    def remove_duplicate_boxes(boxes):
        remove_flag = -1.0
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                if boxes[i].class_index != boxes[j].class_index:
                    continue
                if boxes[j].confidence == remove_flag:
                    continue
                if BoundingBox.iou(boxes[i], boxes[j]) > 0.99:
                    boxes[j].confidence = remove_flag

        new_boxes = []
        for box in boxes:
            if box.confidence != remove_flag:
                new_boxes.append(box)
        return new_boxes

    @staticmethod
    def nms(boxes, iou_threshold=0.45):
        boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
        for i in range(len(boxes) - 1):
            if boxes[i].confidence == 0.0:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j].confidence == 0.0 or boxes[i].class_index != boxes[j].class_index:
                    continue
                if BoundingBox.iou(boxes[i], boxes[j]) > iou_threshold:
                    boxes[j].confidence = 0.0

        nms_filtered_boxes = []
        for box in boxes:
            if box.confidence > 0.0:
                nms_filtered_boxes.append(box)
        return nms_filtered_boxes

    @staticmethod
    def get_color(class_index):
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (212, 255, 127),
            (196, 228, 255),
            (226, 43, 138),
            (42, 42, 165),
            (135, 184, 222),
            (160, 158, 95),
            (0, 255, 127),
            (30, 105, 210),
            (80, 127, 255),
            (237, 149, 100),
            (220, 248, 255),
            (60, 20, 220),
            (255, 255, 0),
            (139, 0, 0),
            (139, 139, 0),
            (11, 134, 184),
            (169, 169, 169),
            (0, 100, 0),
            (169, 169, 169),
            (107, 183, 189),
            (139, 0, 139),
            (47, 107, 85),
            (0, 140, 255),
            (204, 50, 153),
            (0, 0, 139),
            (122, 150, 233),
            (143, 188, 143),
            (139, 61, 72),
            (79, 79, 47),
            (79, 79, 47),
            (209, 206, 0),
            (211, 0, 148),
            (147, 20, 255),
            (255, 191, 0),
            (105, 105, 105),
            (105, 105, 105),
            (255, 144, 30),
            (34, 34, 178),
            (240, 250, 255),
            (34, 139, 34),
            (255, 0, 255),
            (220, 220, 220),
            (255, 248, 248),
            (0, 215, 255),
            (32, 165, 218),
            (128, 128, 128),
            (0, 128, 0),
            (47, 255, 173),
            (128, 128, 128),
            (240, 255, 240),
            (180, 105, 255),
            (92, 92, 205),
            (130, 0, 75),
            (240, 255, 255),
            (140, 230, 240),
            (250, 230, 230),
            (245, 240, 255),
            (0, 252, 124),
            (205, 250, 255),
            (230, 216, 173),
            (128, 128, 240),
            (255, 255, 224),
            (210, 250, 250),
            (211, 211, 211),
            (144, 238, 144),
            (211, 211, 211),
            (193, 182, 255),
            (122, 160, 255),
            (170, 178, 32),
            (250, 206, 135),
            (153, 136, 119),
            (153, 136, 119),
            (222, 196, 176),
            (224, 255, 255),
            (0, 255, 0),
            (50, 205, 50),
            (230, 240, 250),
            (255, 0, 255),
            (0, 0, 128),
            (170, 205, 102),
            (205, 0, 0),
            (211, 85, 186),
            (219, 112, 147),
            (113, 179, 60),
            (238, 104, 123),
            (154, 250, 0),
            (204, 209, 72),
            (133, 21, 199),
            (112, 25, 25),
            (250, 255, 245),
            (225, 228, 255),
            (181, 228, 255),
            (173, 222, 255),
            (128, 0, 0),
            (230, 245, 253),
            (0, 128, 128),
            (35, 142, 107),
            (0, 165, 255),
            (0, 69, 255),
            (214, 112, 218),
            (170, 232, 238),
            (152, 251, 152),
            (238, 238, 175),
            (147, 112, 219),
            (213, 239, 255),
            (185, 218, 255),
            (63, 133, 205),
            (203, 192, 255),
            (221, 160, 221),
            (230, 224, 176),
            (128, 0, 128),
            (0, 0, 255),
            (143, 143, 188),
            (225, 105, 65),
            (19, 69, 139),
            (114, 128, 250),
            (96, 164, 244),
            (87, 139, 46),
            (238, 245, 255),
            (45, 82, 160),
            (192, 192, 192),
            (235, 206, 135),
            (205, 90, 106),
            (144, 128, 112),
            (144, 128, 112),
            (250, 250, 255),
            (127, 255, 0),
            (180, 130, 70),
            (140, 180, 210),
            (128, 128, 0),
            (216, 191, 216),
            (71, 99, 255),
            (208, 224, 64),
            (238, 130, 238),
            (179, 222, 245),
            (255, 255, 255),
            (245, 245, 245),
            (0, 255, 255),
            (50, 205, 154)
        ]
        if class_index < len(colors):
            return colors[class_index]
        else:
            return (255, 255, 255)

