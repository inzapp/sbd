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
import argparse
from yolo import Yolo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/cfg.yaml', help='path of training configuration file')
    parser.add_argument('--gpu', action='store_true', help='use gpu device for model forwarding')
    parser.add_argument('--conf', type=float, default=0.2, help='confidence threshold for detection')
    parser.add_argument('--model', type=str, default='model_last.h5', help='pretrained model path for detection')
    parser.add_argument('--video', type=str, default='', help='video path for detection')
    parser.add_argument('--noclass', action='store_true', help='not showing class label with confidence score')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset name for prediction. train or validation')
    args = parser.parse_args()
    config = Yolo.load_cfg(args.cfg)
    config['pretrained_model_path'] = args.model
    if args.video == '':
        Yolo(config).predict_images(dataset=args.dataset, confidence_threshold=args.conf, device='gpu' if args.gpu else 'cpu', show_class_with_score=not args.noclass)
    else:
        Yolo(config).predict_video(video_path=args.video, confidence_threshold=args.conf, device='gpu' if args.gpu else 'cpu', show_class_with_score=not args.noclass)

