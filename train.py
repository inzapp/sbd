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
        # train_image_path=r'X:\tmp_200m_detection+coco_person\train',
        # validation_image_path=r'X:\tmp_200m_detection+coco_person\validation',
        train_image_path=r'C:\inz\train_data\loon_detection',
        model_name='200m',
        input_shape=(608, 608, 1),
        batch_size=2,
        lr=1e-3,
        epochs=500,
        curriculum_epochs=0,
        validation_split=0.2,
        training_view=False,
        mixed_float16_training=False,
        use_map_callback=True,
        use_lr_scheduler=True,
        lr_scheduler_start_epoch=100,
        lr_scheduler_reduce_factor=0.99)

    # from glob import glob
    # from random import shuffle
    # model = Yolo(pretrained_model_path=r'C:\inz\git\yolo-lab\checkpoints\200m_epoch_4_val_mAP_0.5851.h5', class_names_file_path=r'X:\200m_detection\train\classes.txt')
    # paths = glob(r'C:\inz\videos\200m_detection_2\*.mp4')
    # shuffle(paths)
    # for video_path in paths:
    #     model.predict_video(video_path=video_path)
