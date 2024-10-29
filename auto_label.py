"""
Authors : inzapp

Github url : https://github.com/inzapp/sbd

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

from sbd import SBD, TrainingConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg.yaml', help='path of training configuration file')
    parser.add_argument('--model', type=str, default='auto', help='pretrained model path for detection')
    parser.add_argument('--path', type=str, required=True, default='PATH_WAS_NOT_GIVEN', help='image dir path for auto labeling')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence threshold for detection')
    parser.add_argument('--cpu', action='store_true', help='run with cpu device')
    parser.add_argument('--thresholds', type=str, default='thresholds.txt', help='path of best f1 score confidence thresholds')
    args = parser.parse_args()
    cfg = TrainingConfig(cfg_path=args.cfg)
    cfg.set_config('pretrained_model_path', args.model)
    if args.cpu:
        cfg.set_config('devices', [])
    sbd = SBD(cfg=cfg)
    sbd.auto_label(image_path=args.path, confidence_threshold=args.conf, thresholds_path=args.thresholds)

