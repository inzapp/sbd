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
    'pretrained_model_path': r'',
    'optimizer': 'adam',
    'lr_policy': 'step',
    'model_name': 'loon',
    'alpha': 0.5,
    'gamma': 1.0,
    'label_smoothing': 0.1,
    'train_image_path': r'C:\inz\train_data\loon\train',
    'validation_image_path': r'C:\inz\train_data\loon\validation',
    'class_names_file_path': r'C:\inz\train_data\loon\train\classes.txt',
    'input_shape': (128, 512, 3),

    # 'pretrained_model_path': r'C:\inz\git\yolo-lab\checkpoints\model_3000_iter_mAP_1.0000_f1_1.0000_iou_0.8663_tp_104_fp_0_fn_0_conf_0.8196_ul_all.h5',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'loon',
    # 'alpha': 0.5,
    # 'gamma': 1.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\loon\train',
    # 'validation_image_path': r'C:\inz\train_data\loon\validation',
    # 'class_names_file_path': r'C:\inz\train_data\loon\train\classes.txt',
    # 'input_shape': (128, 512, 3),

    # 'pretrained_model_path': r'./checkpoints/model_2000_iter_mAP_0.7969_f1_0.7962_iou_0.7053_tp_705_fp_153_fn_208_conf_0.4380_ul_all.h5',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',
    # 'model_name': 'lpd',
    # 'alpha': 0.5,
    # 'gamma': 2.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\lp_detection\train',
    # 'validation_image_path': r'C:\inz\train_data\lp_detection\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lp_detection\train\classes.txt',
    # 'input_shape': (384, 640, 1),

    # 'pretrained_model_path': r'C:\inz\git\yolo-lab\checkpoints\lcdw_24000_iter_mAP_0.9647_f1_0.9510_iou_0.8118_tp_8180_fp_599_fn_244_conf_0.8560_ul_all.h5',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',
    # 'model_name': 'lcdw',
    # 'alpha': 0.5,
    # 'gamma': 2.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\lcd_white\train',
    # 'validation_image_path': r'C:\inz\train_data\lcd_white\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lcd_white\train\classes.txt',
    # 'input_shape': (96, 192, 1),

    # 'pretrained_model_path': r'C:\inz\git\yolo-lab\checkpoints\lcd_new\b1\model_129000_iter_mAP_0.9914_f1_0.9923_tp_iou_0.8369_tp_963_fp_6_fn_9_ul_all.h5',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',
    # 'model_name': 'lcdb1',
    # 'alpha': 0.5,
    # 'gamma': 2.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\lcd_new\lcd_b1\train\raw',
    # 'validation_image_path': r'C:\inz\train_data\lcd_new\lcd_b1\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lcd_new\lcd_b1\classes.txt',
    # 'checkpoints': 'checkpoints',
    # 'input_shape': (96, 192, 1),

    # 'pretrained_model_path': r'checkpoints/square/eps_sq_no_1conv_92000_iter_mAP_1.0000_f1_1.0000_tp_iou_0.9955_tp_52_fp_0_fn_0.h5',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',
    # 'model_name': 'square',
    # 'alpha': 0.5,
    # 'gamma': 2.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\square_set',
    # 'validation_image_path': r'C:\inz\train_data\square_set',
    # 'class_names_file_path': r'C:\inz\train_data\square_set\classes.txt',
    # 'input_shape': (640, 640, 1),

    # 'pretrained_model_path': r'checkpoints/lpincar_10000_iter_ul_all.h5',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',
    # 'model_name': 'lpincar',
    # 'alpha': 0.5,
    # 'gamma': 2.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\lp_in_car\train',
    # 'validation_image_path': r'C:\inz\train_data\lp_in_car\validation',
    # 'class_names_file_path': r'C:\inz\train_data\lp_in_car\train\classes.txt',
    # 'input_shape': (256, 256, 1),

    # 'pretrained_model_path': r'C:\inz\git\yolo-lab\checkpoints\normalsub_24000_iter_mAP_0.5221_f1_0.7186_iou_0.8143_tp_2600_fp_598_fn_1438_conf_0.4033.h5',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',
    # 'model_name': 'normalsub',
    # 'alpha': 0.5,
    # 'gamma': 2.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\normal_model_1028\train_subset',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_1028\validation_subset',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_1028\train\classes.txt',
    # 'input_shape': (384, 640, 1),

    # 'pretrained_model_path': r'C:\inz\git\yolo-lab\checkpoints\normalcarsub_12000_iter_mAP_0.1536_f1_0.6640_iou_0.7741_tp_251_fp_96_fn_158_conf_0.2026.h5',
    # 'optimizer': 'sgd',
    # 'lr_policy': 'step',
    # 'model_name': 'normalcarsub',
    # 'alpha': 0.5,
    # 'gamma': 2.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train_subset',
    # 'validation_image_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\validation_subset',
    # 'class_names_file_path': r'C:\inz\train_data\normal_model_car_640x384_12_class\train\classes.txt',
    # 'input_shape': (384, 640, 1),

    # 'pretrained_model_path': r'C:\inz\git\yolo-lab\checkpoints\normalcarsub_12000_iter_mAP_0.1536_f1_0.6640_iou_0.7741_tp_251_fp_96_fn_158_conf_0.2026.h5',
    # 'optimizer': 'adam',
    # 'lr_policy': 'step',
    # 'model_name': 'covid',
    # 'alpha': 0.5,
    # 'gamma': 1.0,
    # 'label_smoothing': 0.1,
    # 'train_image_path': r'C:\inz\train_data\kaggle\covid-detection\jpg\origin_1024x1024\train',
    # 'validation_image_path': r'C:\inz\train_data\kaggle\covid-detection\jpg\origin_1024x1024\test',
    # 'class_names_file_path': r'C:\inz\train_data\kaggle\covid-detection\jpg\origin_1024x1024\train\classes.txt',
    # 'input_shape': (224, 224, 1),

    'lr': 1e-3,
    'decay': 0.0005,
    'momentum': 0.9,
    'burn_in': 1000,
    'batch_size': 4,
    'iterations': 1000000,
    'curriculum_iterations': 0,
    'checkpoints': 'checkpoint',
    'multi_classification_at_same_box': False,
    'map_checkpoint': True,
    'training_view': True
}

