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
    parser.add_argument('--opset', type=int, default=12, help='onnx export opset')
    args = parser.parse_args()
    cfg = TrainingConfig(cfg_path=args.cfg)
    cfg.set_config('pretrained_model_path', args.model)
    sbd = SBD(cfg=cfg)
    sbd.export(opset=args.opset)

