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
    parser.add_argument('--conf', type=float, default=0.2, help='confidence threshold for detection')
    parser.add_argument('--path', type=str, default='', help='image or video path for detection')
    parser.add_argument('--width', type=int, default=0, help='width for showing detection result')
    parser.add_argument('--height', type=int, default=0, help='height for showing detection result')
    parser.add_argument('--hide-class', action='store_true', help='not showing class label with confidence score')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset name for prediction. train or validation')
    parser.add_argument('--heatmap', action='store_true', help='show objectness heatmap blended image')
    parser.add_argument('--thresholds', type=str, default='thresholds.txt', help='path of best f1 score confidence thresholds')
    args = parser.parse_args()
    cfg = TrainingConfig(cfg_path=args.cfg)
    cfg.set_config('pretrained_model_path', args.model)
    sbd = SBD(cfg=cfg)
    sbd.detect(
        path=args.path,
        dataset=args.dataset,
        confidence_threshold=args.conf,
        show_class=not args.hide_class,
        width=args.width,
        height=args.height,
        heatmap=args.heatmap,
        thresholds_path=args.thresholds)

