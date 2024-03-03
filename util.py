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
import cv2
import numpy as np
import tensorflow as tf


class Util:
    def __init__(self):
        pass

    @staticmethod
    def print_error_exit(msg):
        msg_type = type(msg)
        if msg_type is str:
            msg = [msg]
        msg_type = type(msg)
        if msg_type is list:
            print()
            for s in msg:
                print(f'[ERROR] {s}')
        else:
            print(f'[print_error_exit] msg print failure. invalid msg type : {msg_type}')
        exit(-1)

