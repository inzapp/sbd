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

config = {
    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'loon',
    # 'train_image_path': r'C:\inz\train_data\loon\train',
    # 'validation_image_path': r'C:\inz\train_data\loon\validation',
    # 'class_names_file_path': r'C:\inz\train_data\loon\train\classes.txt',
    # 'input_shape': (128, 512, 3),

    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'lpd',
    # 'train_image_path': r'C:\inz\train_data\lp_detection\train',
    # 'validation_image_path': r'C:\inz\train_data\lp_detection\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lp_detection\train\classes.txt',
    # 'input_shape': (384, 640, 1),

    'pretrained_model_path': r'',
    'optimizer': 'sgd',
    'lr_policy': 'step',
    'model_name': 'lcdw',
    'train_image_path': r'C:\inz\train_data\lcd_white\train',
    'validation_image_path': r'C:\inz\train_data\lcd_white\validation',
    'class_names_file_path': r'C:\inz\train_data\lcd_white\train\classes.txt',
    'input_shape': (96, 192, 1),

    # 'pretrained_model_path': r'C:\inz\git\yolo-lab\checkpoints\lcd_new\b1\model_129000_iter_mAP_0.9914_f1_0.9923_tp_iou_0.8369_tp_963_fp_6_fn_9_ul_all.h5',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',
    # 'model_name': 'lcdb1',
    # 'train_image_path': r'C:\inz\train_data\lcd_new\lcd_b1\train\raw',
    # 'validation_image_path': r'C:\inz\train_data\lcd_new\lcd_b1\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lcd_new\lcd_b1\classes.txt',
    # 'checkpoints': 'checkpoints',
    # 'input_shape': (96, 192, 1),

    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'square',
    # 'train_image_path': r'C:\inz\train_data\square_set',
    # 'validation_image_path': r'C:\inz\train_data\square_set',
    # 'class_names_file_path': r'C:\inz\train_data\square_set\classes.txt',
    # 'input_shape': (640, 640, 1),

    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'lpincar',
    # 'train_image_path': r'C:\inz\train_data\lp_in_car\train',
    # 'validation_image_path': r'C:\inz\train_data\lp_in_car\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lp_in_car\train\classes.txt',
    # 'input_shape': (256, 256, 1),

    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'normal',
    # 'train_image_path': r'C:\inz\train_data\normal_model_1028\train',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_1028\validation',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_1028\train\classes.txt',
    # 'input_shape': (384, 640, 1),

    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'normalsub',
    # 'train_image_path': r'C:\inz\train_data\normal_model_1028\train_subset',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_1028\validation_subset',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_1028\train\classes.txt',
    # 'input_shape': (384, 640, 1),

    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'normalcar',
    # 'train_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train_256x128',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\validation_256x128',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train_256x128\classes.txt',
    # 'input_shape': (128, 256, 1),

    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'normalcarsub',
    # 'train_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train_subset',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\validation_subset',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train\classes.txt',
    # 'input_shape': (384, 640, 1),

    # 'pretrained_model_path': r'',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'kagglecar',
    # 'train_image_path': r'C:\inz\train_data\kaggle\car_detection\train\done',
    # 'validation_image_path': r'C:\inz\train_data\kaggle\car_detection\validation',
    # 'class_names_file_path': r'C:\inz\train_data\kaggle\car_detection\train\classes.txt',
    # 'input_shape': (256, 512, 1),

    'lr': 1e-3,
    'l2': 5e-4,
    'warm_up': 0.5,
    'decay_step': 0.25,
    'momentum': 0.9,
    'alpha': 0.25,
    'gamma': 2.0,
    'label_smoothing': 0.1,
    'batch_size': 4,
    'iterations': 10000,
    'curriculum_iterations': 0,
    'checkpoint_path': 'checkpoint',
    'multi_classification_at_same_box': False,
    'map_checkpoint': True,
    'training_view': False 
}

