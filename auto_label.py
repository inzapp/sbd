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

