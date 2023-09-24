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
from sbd import SBD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg.yaml', help='path of training configuration file')
    parser.add_argument('--model', type=str, default='best.h5', help='pretrained model path for detection')
    parser.add_argument('--save', action='store_true', help='save another model with calculated mAP result naming')
    parser.add_argument('--cached', action='store_true', help='use pre-saved csv files for mAP calculation')
    parser.add_argument('--find-best-threshold', action='store_true', help='find best confidence threshold for f1 score')
    parser.add_argument('--iou', type=float, default=0.5, help='true positive threshold for intersection over union')
    parser.add_argument('--conf', type=float, default=0.2, help='confidence threshold for detection')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset name for mAP calculation. train or validation')
    parser.add_argument('--truecsv', type=str, default='annotations.csv', help='annotations csv path for cached mAP calculation')
    parser.add_argument('--predcsv', type=str, default='predictions.csv', help='predictions csv path for cached mAP calculation')
    args = parser.parse_args()
    sbd = SBD(cfg_path=args.cfg, training=False)
    sbd.load_model(args.model)
    sbd.calculate_map(
        dataset=args.dataset,
        confidence_threshold=args.conf,
        tp_iou_threshold=args.iou,
        cached=args.cached,
        find_best_threshold=args.find_best_threshold,
        annotations_csv_path=args.truecsv,
        predictions_csv_path=args.predcsv)

