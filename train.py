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
from glob import glob

if __name__ == '__main__':
    """
    Train model using fit method.
    
    train_image_path:
        The path to the directory where the training data is located.
        
        There must be images and labels in this directory.
        The image and label must have the same file name and not be in different directories.
        
    validation_image_path:
        Use this parameter if the validation data is in a different path from the training data.
        When this parameter is used, validation split is ignored.
        
    input_shape:
        (height, width, channels) format of model input size
        If the channel is 1, train with a gray image, otherwise train with a color image.
        
    batch_size:
        batch size of training.
        
    lr:
        Learning rate value while training. 1e-3 is good for most cases.
        
    decay:
        L2 weight decay regularizing parameter.
        
    momentum:
        The exponential decay rate for the 1st moment estimates.
        beta_1 parameter value for Adam optimizer.
        
    burn_in:
        Warming up iteration count before train using lr parameter. 1000 is good for most cases.
        
    iterations:
        Total training iteration count.
        
    curriculum_iterations:
        Iterations to pre-reduce the loss to the confidence and bounding box channel before starting the training.
        curriculum_iterations = burn_in * 2 + given value
        
    validation_split:
        The percentage of data that will be randomly used as validation data. default value is 0.2
        
    training_view:
        During training, the image is forwarded in real time, showing the results are shown.
        
    mixed_float16_training:
        Use of both 16-bit and 32-bit floating point types in the model during training to run faster and use less memory.
    """

    Yolo(
        # pretrained_model_path=r'checkpoints/model_5000_iter_mAP_1.0000_f1_0.9709_tp_iou_0.8272_tp_100_fp_2_fn_4.h5',
        # optimizer='sgd',
        # lr_policy='step',
        # train_image_path=r'C:\inz\train_data\loon\train',
        # validation_image_path=r'C:\inz\train_data\loon\validation',
        # class_names_file_path=r'C:\inz\train_data\loon\train\classes.txt',
        # input_shape=(128, 512, 3),

        # train_image_path=r'C:\inz\train_data\200m_sequence_for_auto_label\train',
        # validation_image_path=r'C:\inz\train_data\200m_sequence_for_auto_label\validation',
        # class_names_file_path=r'C:\inz\train_data\200m_sequence_for_auto_label\train\classes.txt',
        # input_shape=(384, 640, 1),

        # pretrained_model_path=r'checkpoints\small_drop_filter_training\model_100000_iter_mAP_0.4574_f1_0.6522_tp_iou_0.8082.h5',
        # train_image_path=r'list\small_train.txt',
        # validation_image_path=r'list\small_validation.txt',
        # class_names_file_path=r'T:\200m_big_small_detection\train_data\small\small_all\train\classes.txt',
        # input_shape=(192, 576, 1),

        # pretrained_model_path=r'checkpoints/person/head_detail/model_722000_iter_mAP_0.6116_f1_0.5790_tp_iou_0.7655_tp_1954_fp_794_fn_2048.h5',
        # optimizer='adam',
        # train_image_path=r'C:\inz\train_data\lp_detection\train',
        # validation_image_path=r'C:\inz\train_data\lp_detection\validation',
        # class_names_file_path=r'C:\inz\train_data\lp_detection\train\classes.txt',
        # input_shape=(384, 640, 1),

        # pretrained_model_path=r'checkpoints/object_detail/head_detail/model_722000_iter_mAP_0.6116_f1_0.5790_tp_iou_0.7655_tp_1954_fp_794_fn_2048.h5',
        # optimizer='sgd',
        # lr_policy='cosine',
        # train_image_path=r'list/head_crop_detection/train.txt',
        # validation_image_path=r'list/head_crop_detection/validation.txt',
        # class_names_file_path=r'C:\inz\train_data\head_crop_detail\train\classes.txt',
        # input_shape=(128, 128, 1),

        # pretrained_model_path=r'checkpoints/lcd_white/model_32000_iter_mAP_0.8227_f1_0.9569_tp_iou_0.8338.h5',
        # train_image_path=r'C:\inz\train_data\lcd_white\train',
        # validation_image_path=r'C:\inz\train_data\lcd_white\validation',
        # class_names_file_path=r'C:\inz\train_data\lcd_white\train\classes.txt',
        # input_shape=(96, 192, 1),

        # pretrained_model_path=r'checkpoints/model_6000_iter_mAP_0.8519_f1_0.9412_tp_iou_0.6763_tp_8_fp_0_fn_1.h5',
        optimizer='sgd',
        lr_policy='step',
        train_image_path=r'C:\inz\train_data\square_set',
        validation_image_path=r'C:\inz\train_data\square_set',
        class_names_file_path=r'C:\inz\train_data\square_set\classes.txt',
        # input_shape=(128, 128, 1),
        input_shape=(96, 96, 1),

        lr=0.001,
        decay=0.0005,
        momentum=0.9,
        burn_in=1000,
        batch_size=2,
        iterations=10000,
        curriculum_iterations=0,
        training_view=True,
        map_checkpoint=True).fit()
        # map_checkpoint=True).predict_validation_images()
        # map_checkpoint=True).map_validation_images()

    # from glob import glob
    # from random import shuffle
    # model_path = r'C:\inz\git\yolo-lab\checkpoints\model_66000_iter_mAP_0.8389_f1_0.8627_tp_iou_0.8111.h5'
    # model = Yolo(pretrained_model_path=model_path, class_names_file_path=r'X:\200m_detection\origin\train\classes.txt', test_only=True)
    # paths = glob(r'Z:\07. SW 개발팀\08. 개인 폴더\24. 표성백\22.돌발상황감지\2021-05-17\*.mp4')
    # shuffle(paths)
    # for video_path in paths:
    #     model.predict_video(video_path)
