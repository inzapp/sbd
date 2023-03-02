"""
Authors : inzapp

Github url : https://github.com/inzapp/c-yolo

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from yolo import Yolo


config = {
    'pretrained_model_path': r'',
    'model_name': 'loon',
    'model_type': 'x1',
    'input_shape': (128, 512, 3),
    'train_image_path': r'C:\inz\train_data\loon\train',
    'validation_image_path': r'C:\inz\train_data\loon\validation',
    'class_names_file_path': r'C:\inz\train_data\loon\train\classes.txt',
    'optimizer': 'sgd',
    'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'lpd',
    # 'model_type': 's1',
    # 'input_shape': (384, 640, 1),
    # 'train_image_path': r'C:\inz\train_data\lp_detection\train',
    # 'validation_image_path': r'C:\inz\train_data\lp_detection\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lp_detection\train\classes.txt',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'lcdw',
    # 'model_type': 'n1',
    # 'input_shape': (96, 192, 1),
    # 'train_image_path': r'C:\inz\train_data\lcd_white\train',
    # 'validation_image_path': r'C:\inz\train_data\lcd_white\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lcd_white\train\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'lcdb1',
    # 'model_type': 'lcd',
    # 'input_shape': (96, 192, 1),
    # 'train_image_path': r'C:\inz\train_data\lcd_new\lcd_b1\train\raw',
    # 'validation_image_path': r'C:\inz\train_data\lcd_new\lcd_b1\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lcd_new\lcd_b1\classes.txt',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'square',
    # 'model_type': 's1',
    # 'input_shape': (320, 320, 1),
    # 'train_image_path': r'C:\inz\train_data\square_set',
    # 'validation_image_path': r'C:\inz\train_data\square_set',
    # 'class_names_file_path': r'C:\inz\train_data\square_set\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'lpincar',
    # 'model_type': 'n1',
    # 'input_shape': (256, 256, 1),
    # 'train_image_path': r'C:\inz\train_data\lp_in_car\train',
    # 'validation_image_path': r'C:\inz\train_data\lp_in_car\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lp_in_car\train\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'normal',
    # 'model_type': 'l1',
    # 'input_shape': (384, 640, 1),
    # 'train_image_path': r'C:\inz\train_data\normal_model_1028\train',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_1028\validation',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_1028\train\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'normalsub',
    # 'model_type': 's1',
    # 'input_shape': (384, 640, 1),
    # 'train_image_path': r'C:\inz\train_data\normal_model_1028\train_subset',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_1028\validation_subset',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_1028\train_subset\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'normalcar',
    # 'model_type': 's1',
    # 'input_shape': (128, 256, 1),
    # 'train_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train_256x128',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\validation_256x128',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train_256x128\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_name': 'normalcarsub',
    # 'model_type': 's1',
    # 'input_shape': (384, 640, 1),
    # 'train_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train_subset',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\validation_subset',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train\classes.txt',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_type': 's1',
    # 'model_name': 'kagglecar',
    # 'input_shape': (256, 512, 1),
    # # 'input_shape': (384, 640, 1),
    # 'train_image_path': r'C:\inz\train_data\kaggle\car_detection\train',
    # 'validation_image_path': r'C:\inz\train_data\kaggle\car_detection\validation',
    # 'class_names_file_path': r'C:\inz\train_data\kaggle\car_detection\train\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_type': 'x1',
    # 'model_name': 'normal_19cls',
    # 'input_shape': (384, 640, 1),
    # 'train_image_path': r'C:\inz\train_data\normal_model_20230213_19cls\train',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_20230213_19cls\validation',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_20230213_19cls\train\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_type': 'n1',
    # 'model_name': 'lp_car_sub',
    # 'input_shape': (352, 640, 1),
    # 'train_image_path': r'C:\inz\train_data\lp_car_detection_sub\train',
    # 'validation_image_path': r'C:\inz\train_data\lp_car_detection_sub\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lp_car_detection_sub\train\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    # 'pretrained_model_path': r'',
    # 'model_type': 'n1',
    # 'model_name': 'lp_car_with_normal',
    # 'input_shape': (352, 640, 1),
    # 'train_image_path': r'C:\inz\train_data\lp_car_detection_with_normal_640x384\train',
    # 'validation_image_path': r'C:\inz\train_data\lp_car_detection_with_normal_640x384\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lp_car_detection_with_normal_640x384\train\classes.txt',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',

    'lr': 1e-3,
    'l2': 5e-4,
    'alpha': 0.25,
    'gamma': 1.0,
    'momentum': 0.9,
    'warm_up': 0.1,
    'decay_step': 0.1,
    'label_smoothing': 0.0,
    'batch_size': 4,
    'iterations': 10000,
    'curriculum_iterations': 0,
    'checkpoint_interval': 1000,
    'checkpoint_path': 'checkpoint',
    'ignore_nearby_cell': False,
    'nearby_cell_ignore_threshold': 0.5,
    'multi_classification_at_same_box': False,
    'map_checkpoint': True,
    'training_view': False
}

if __name__ == '__main__':
    Yolo(config=config).fit()

