"""
Authors : inzapp

Github : https://github.com/inzapp/sbd
"""
import argparse

from sbd import SBD, TrainingConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/cfg.yaml', help='path of training configuration file')
    parser.add_argument('--model', type=str, default=None, help='pretrained model path')
    parser.add_argument('--show-progress', action='store_true', help='show training progress with live prediction')
    args = parser.parse_args()
    cfg = TrainingConfig(cfg_path=args.cfg)
    if args.show_progress!= '':
        cfg.set_config('show_progress', args.show_progress)
    if args.model != '':
        cfg.set_config('pretrained_model_path', args.model)
    SBD(cfg=cfg).train()

