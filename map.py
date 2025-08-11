"""
Authors : inzapp

Github : https://github.com/inzapp/sbd
"""
import argparse

from sbd import SBD, TrainingConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg.yaml', help='path of training configuration file')
    parser.add_argument('--model', type=str, default='auto', help='pretrained model path for detection')
    parser.add_argument('--cached', action='store_true', help='use pre-saved csv files for mAP calculation')
    parser.add_argument('--iou', type=float, default=0.5, help='true positive threshold for intersection over union')
    parser.add_argument('--conf', type=float, default=0.0, help='confidence threshold for detection, 0 for calculating best f1 threshold')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset name for mAP calculation. train or validation')
    parser.add_argument('--truecsv', type=str, default='annotations.csv', help='annotations csv path for cached mAP calculation')
    parser.add_argument('--predcsv', type=str, default='predictions.csv', help='predictions csv path for cached mAP calculation')
    args = parser.parse_args()
    cfg = TrainingConfig(cfg_path=args.cfg)
    cfg.set_config('pretrained_model_path', args.model)
    sbd = SBD(cfg=cfg)
    sbd.evaluate(
        dataset=args.dataset,
        confidence_threshold=args.conf,
        tp_iou_threshold=args.iou,
        cached=args.cached,
        annotations_csv_path=args.truecsv,
        predictions_csv_path=args.predcsv)

