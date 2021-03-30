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

if __name__ == '__main__':
    """
    Train model using fit method.
    
    train_image_path:
        The path to the directory where the training data is located.
        
        There must be images and labels in this directory.
        The image and label must have the same file name and not be in different directories.
        
    input_shape:
        (height, width, channel) format of model input size
        If the channel is 1, train with a gray image, otherwise train with a color image.
        
    batch_size:
        2 batch is recommended.
        
    lr:
        Learning rate value while training. 1e-3 ~ 1e-4 is recommended.
        
    epochs:
        Epochs value.
        
    curriculum_epochs:
        Epochs to pre-reduce the loss to the confidence and bounding box channel before starting the training.
        
    validation_split:
        The percentage of data that will be used as validation data.
        
    validation_image_path:
        Use this parameter if the validation data is in a different path from the training data.
        
    training_view:
        During training, the image is forwarded in real time, showing the results are shown.
        False if training is on a server system without IO equipment.
    """

    # model = Yolo()
    # model.fit(
    #     train_image_path=r'\\192.168.101.200\train_data\person_data_train',
    #     validation_image_path=r'\\192.168.101.200\train_data\person_data_validation',
    #     model_name='person_info_detector',
    #     input_shape=(256, 128, 3),
    #     batch_size=2,
    #     lr=1e-3,
    #     epochs=300,
    #     curriculum_epochs=0,
    #     validation_split=0.2,
    #     training_view=False,
    #     mixed_float16_training=False,
    #     use_map_callback=True)

    model = Yolo()
    model.fit(
        train_image_path=r'\\192.168.101.200\train_data\lp_detection_train',
        validation_image_path=r'\\192.168.101.200\train_data\lp_detection_validation',
        model_name='v2_sbd',
        input_shape=(368, 640, 3),
        batch_size=2,
        lr=1e-3,
        epochs=300,
        curriculum_epochs=5,
        validation_split=0.0,
        training_view=False,
        mixed_float16_training=False,
        use_map_callback=True)

    # model = Yolo(pretrained_model_path=r'person_detector_model_epoch_17_f1_0.5952_val_f1_0.1739.h5', class_names_file_path=r'X:\person_data\classes.txt')
    # from glob import glob
    # model.predict_images(glob(r'X:\person_data\*.jpg'))
