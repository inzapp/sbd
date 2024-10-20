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

