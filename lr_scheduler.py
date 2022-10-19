"""
Authors : inzapp

Github url : https://github.com/inzapp/c-yolo

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
import numpy as np


class LRScheduler:
    def __init__(self,
                 iterations,
                 lr,
                 min_lr=0.0,
                 min_momentum=0.85,
                 max_momentum=0.95,
                 initial_cycle_length=2500,
                 cycle_weight=2):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = self.lr
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.iterations = iterations
        self.cycle_length = initial_cycle_length
        self.cycle_weight = cycle_weight
        self.cycle_step = 0

    def update(self, optimizer, iteration_count, burn_in, lr_policy):
        if lr_policy == 'step':
            self.__schedule_step_decay(optimizer, iteration_count, burn_in=burn_in)
        elif lr_policy == 'cosine':
            self.__schedule_cosine_warm_restart(optimizer, iteration_count, burn_in=burn_in)
        elif lr_policy == 'onecycle':
            self.__schedule_one_cycle(optimizer, iteration_count)
        elif lr_policy == 'constant':
            pass
        else:
            print(f'{lr_policy} is invalid lr policy')
            return

    def __set_lr(self, optimizer, lr):
        optimizer.__setattr__('lr', lr)

    def __set_momentum(self, optimizer, momentum):
        attr = ''
        optimizer_str = optimizer.__str__().lower()
        if optimizer_str.find('sgd') > -1:
            optimizer.__setattr__('momentum', momentum)
        elif optimizer_str.find('adam') > -1:
            optimizer.__setattr__('beta_1', momentum)

    def __burn_in_lr(self, iteration_count, burn_in):
        return self.lr * pow(iteration_count / float(burn_in), 4)

    def __schedule_step_decay(self, optimizer, iteration_count, burn_in=1000):
        if burn_in > 0 and iteration_count <= burn_in:
            lr = self.__burn_in_lr(iteration_count, burn_in)
        elif iteration_count == int(self.iterations * 0.8):
            lr = self.lr * 0.1
        elif iteration_count == int(self.iterations * 0.9):
            lr = self.lr * 0.01
        else:
            lr = self.lr
        self.__set_lr(optimizer, lr)

    def __schedule_one_cycle(self, optimizer, iteration_count):
        warm_up = 0.1
        min_lr = self.min_lr
        max_lr = self.max_lr
        min_mm = self.min_momentum
        max_mm = self.max_momentum
        warm_up_iterations = int(self.iterations * warm_up)
        if iteration_count <= warm_up_iterations:
            iterations = warm_up_iterations
            lr = ((np.cos(((iteration_count * np.pi) / iterations) + np.pi) + 1.0) * 0.5) * (max_lr - min_lr) + min_lr  # increase only until target iterations
            mm = ((np.cos(((iteration_count * np.pi) / iterations) +   0.0) + 1.0) * 0.5) * (max_mm - min_mm) + min_mm  # decrease only until target iterations
            self.__set_lr(optimizer, lr)
            self.__set_momentum(optimizer, mm)
        else:
            iteration_count -= warm_up_iterations + 1
            iterations = self.iterations - warm_up_iterations
            lr = ((np.cos(((iteration_count * np.pi) / iterations) +   0.0) + 1.0) * 0.5) * (max_lr - min_lr) + min_lr  # decrease only until target iterations
            mm = ((np.cos(((iteration_count * np.pi) / iterations) + np.pi) + 1.0) * 0.5) * (max_mm - min_mm) + min_mm  # increase only until target iterations
            self.__set_lr(optimizer, lr)
            self.__set_momentum(optimizer, mm)

    def __schedule_cosine_warm_restart(self, optimizer, iteration_count, burn_in=1000):
        if burn_in > 0 and iteration_count <= burn_in:
            lr = self.__burn_in_lr(iteration_count, burn_in)
        else:
            if self.cycle_step % self.cycle_length == 0 and self.cycle_step != 0:
                self.cycle_step = 0
                self.cycle_length = int(self.cycle_length * self.cycle_weight)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(((1.0 / self.cycle_length) * np.pi * (self.cycle_step % self.cycle_length))))  # down and down
            self.cycle_step += 1
        self.__set_lr(optimizer, lr)

