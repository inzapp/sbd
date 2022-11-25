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
from train_config import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='forward with cpu while calculating mAP')
    parser.add_argument('--save', action='store_true', help='save another model with calculated mAP result naming')
    parser.add_argument('--cached', action='store_true', help='use pre-saved csv files for mAP calculation')
    parser.add_argument('--iou', type=float, default=0.5, help='true positive threshold for intersection over union')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold for detection')
    parser.add_argument('--model', type=str, default='model_last.h5', help='pretrained model path for mAP calculation')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset name for mAP calculation. train or validation')
    args = parser.parse_args()
    config['pretrained_model_path'] = args.model
    Yolo(config=config).calculate_map(args.dataset, save_model=args.save, device='cpu' if args.cpu else 'auto', confidence_threshold=args.conf, tp_iou_threshold=args.iou, cached=args.cached)

