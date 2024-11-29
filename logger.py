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


class Logger:
    def __init__(self):
        pass

    @staticmethod
    def log(msg, header, end='\n'):
        msg_type = type(msg)
        assert msg_type in [str, list]
        if msg_type is str:
            print(f'{header} {msg}', end=end)
        else:
            for i, s in enumerate(msg):
                if i == 0:
                    print(f'{header} {s}')
                else:
                    print(f'    {s}')

    @staticmethod
    def info(msg, end='\n'):
        Logger.log(msg, header='\033[1;32m[INFO]\033[0m', end=end)

    @staticmethod
    def warn(msg, end='\n'):
        Logger.log(msg, header='\033[1;33m[WARNING]\033[0m', end=end)

    @staticmethod
    def error(msg, end='\n', callback=None):
        Logger.log(msg, header='\033[1;31m[ERROR]\033[0m', end=end)
        if callback:
            callback()
        exit(-1)

