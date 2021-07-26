import sys
from glob import glob

import cv2

g_win_name = 'Label RGB v1.0 by Inzapp'


def mouse_callback(event, cur_x, cur_y, flag, _):
    global g_raw

    # no click mouse moving
    if event == 0 and flag == 0:
        raw_height, raw_width = g_raw.shape[0], g_raw.shape[1]
        y_ratio = cur_y / raw_height
        x_ratio = cur_x / raw_width
        print(f'x : {x_ratio}')
        print(f'y : {y_ratio}')

    # end click
    elif event == 4 and flag == 0:
        pass


path = ''
if len(sys.argv) > 1:
    path = sys.argv[1].replace('\\', '/') + '/'

jpg_file_paths = glob(f'{path}*.jpg')
png_file_paths = glob(f'{path}*.png')
img_paths = jpg_file_paths + png_file_paths
if len(img_paths) == 0:
    print('No image files in path. run label.py with path argument')
    sys.exit(0)

index = 0
while True:
    file_path = img_paths[index]
    g_label_path = f'{file_path[:-4]}.txt'
    g_raw = cv2.imread(file_path, cv2.IMREAD_COLOR)
    cv2.namedWindow(g_win_name)
    cv2.imshow(g_win_name, g_raw)
    cv2.setMouseCallback(g_win_name, mouse_callback)
    res = cv2.waitKey(0)

    # go to next if input key was 'd'
    if res == ord('d'):
        if index == len(img_paths) - 1:
            print('Current image is last image')
        else:
            index += 1

    # go to previous image if input key was 'a'
    elif res == ord('a'):
        if index == 0:
            print('Current image is first image')
        else:
            index -= 1

    # exit if input key was ESC
    elif res == 27:
        sys.exit()
