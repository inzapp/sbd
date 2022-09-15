import os
from glob import glob


g_train_dir_paths = [
    r'Z:\kgu\TOTAL\result\train',
]

g_validation_dir_paths = [
    r'Z:\kgu\TOTAL\result\validation',
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


def main():
    global g_train_dir_paths, g_validation_dir_paths 
    make_paths(g_train_dir_paths, 'train.txt')
    make_paths(g_validation_dir_paths, 'validation.txt')


if __name__ == '__main__':
    main()
