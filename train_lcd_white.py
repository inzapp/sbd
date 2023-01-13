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
    'model_name': 'lcd_white',
    'model_type': 'lcd',
    'input_shape': (96, 192, 1),
    'train_image_path': r'/home2/train_data_200_server/03_LPR/02_LPR/lpr/lp_character_detection/clean_data/lcd_white/train',
    'validation_image_path': r'/home2/train_data_200_server/03_LPR/02_LPR/lpr/lp_character_detection/clean_data/lcd_white/validation',
    'class_names_file_path': r'/home2/train_data_200_server/03_LPR/02_LPR/lpr/lp_character_detection/clean_data/lcd_white/classes.txt',
    'optimizer': 'adam',
    'lr_policy': 'step',
    'lr': 1e-3,
    'l2': 5e-4,
    'alpha': 0.25,
    'gamma': 2.0,
    'momentum': 0.9,
    'warm_up': 0.5,
    'decay_step': 0.1,
    'label_smoothing': 0.1,
    'batch_size': 4,
    'iterations': 30000,
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

