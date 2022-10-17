import os
import shutil as sh

from glob import glob
from tqdm import tqdm
from concurrent.futures.thread import ThreadPoolExecutor


# g_train_dir_paths = [
#     r'Z:\kgu\TOTAL\result\train',
# ]
# 
# g_validation_dir_paths = [
#     r'Z:\kgu\TOTAL\result\validation',
# ]

g_train_dir_paths = [
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/200m_detection/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/20220117_1F/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/20220118_balcony_s1/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/20220119_balcony_s2/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/20220120_balcony_s3/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/20220125_bikes/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/20220125_person_selected/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/cat/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/CDF_220407/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/coco/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/coco2017_cat_dog/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/deer/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/deer2/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/detail_detection/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/lp_image/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/pig/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/stroller/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/surveillance/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/surveillance_night/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/surveillance2/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/youtube_7_class/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/youtube_bike/train',
    r'//192.168.101.200/home2/jysung/08_FACE/CV2_NORMAL_12cls/youtube_street/train',
    r'//192.168.101.200/home2/jysung/05_CV2_NORMAL/CV2_NORMAL_12class_head_revised/background/empty_street',
]

g_validation_dir_paths = [
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/200m_detection/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/20220117_1F/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/20220118_balcony_s1/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/20220119_balcony_s2/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/20220120_balcony_s3/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/20220125_bikes/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/20220125_person_selected/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/cat/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/CDF_220407/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/coco/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/coco2017_cat_dog/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/deer/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/deer2/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/detail_detection/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/lp_image/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/pig/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/stroller/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/surveillance/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/surveillance_night/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/surveillance2/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/youtube_7_class/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/youtube_bike/validation',
    r'//192.168.101.200/home2/jysung//08_FACE/CV2_NORMAL_12cls/youtube_street/validation',
]


def make_paths(dir_paths, file_name):
    paths = []
    for dir_path in dir_paths:
        paths += glob(f'{dir_path}/**/*.jpg', recursive=True)
    for i in range(len(paths)):
        paths[i] = f'{paths[i]}\n'
    with open(file_name, 'wt') as f:
        f.writelines(paths)
    print(file_name)
    print(f'image count : {len(paths)}')


def copy_if_exists(file_name, dst_dir):
    if os.path.exists(file_name) and os.path.isfile(file_name):
        sh.copy(file_name, dst_dir)


def copy_paths(file_name, dst_dir):
    if not (os.path.exists(dst_dir) and os.path.isdir(dst_dir)):
        os.makedirs(dst_dir, exists_ok=True)
    with open(file_name, 'rt') as f:
        paths = f.readlines()
    fs = []
    pool = ThreadPoolExecutor(16)
    for path in tqdm(paths):
        path = path.replace('\n', '')
        label_path = f'{path[:-4]}.txt'
        fs.append(pool.submit(copy_if_exists, path, dst_dir))
        fs.append(pool.submit(copy_if_exists, label_path, dst_dir))
    for f in tqdm(fs):
        f.result()


def main():
    global g_train_dir_paths, g_validation_dir_paths 
    make_paths(g_train_dir_paths, 'train.txt')
    make_paths(g_validation_dir_paths, 'validation.txt')
    # copy_paths('./list/normal_car_detection/train.txt', r'C:\inz\train_data\normal_car\train')
    # copy_paths('./list/normal_car_detection/validation.txt', r'C:\inz\train_data\normal_car\validation')


if __name__ == '__main__':
    main()

