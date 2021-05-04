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

    model = Yolo()
    model.fit(
        train_image_path=r'X:\person\face_helmet_added\train',
        validation_image_path=r'X:\person\face_helmet_added\validation',
        model_name='sgd_v2_person_info_detector_192_96',
        input_shape=(192, 96, 1),
        batch_size=2,
        lr=1e-3,
        epochs=500,
        curriculum_epochs=5,
        validation_split=0.0,
        training_view=True,
        mixed_float16_training=True,
        use_map_callback=True,
        use_lr_scheduler=True,
        lr_scheduler_start_epoch=100,
        lr_scheduler_reduce_factor=0.98)

    # from glob import glob
    # model = Yolo(pretrained_model_path=r'C:\inz\git\yolo-lab\checkpoints\v2_person_info_detector_192_96_epoch_19_val_mAP_0.2353.h5', class_names_file_path=r'X:\person\face_helmet_added\validation\classes.txt')
    # model.predict_images(glob(r'X:\person\face_helmet_added\validation\*.jpg'))
