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

    def update(self, optimizer, iteration_count, warm_up, lr_policy):
        if lr_policy == 'step':
            lr = self.__schedule_step_decay(optimizer, iteration_count, warm_up=warm_up)
        elif lr_policy == 'cosine':
            lr = self.__schedule_cosine_warm_restart(optimizer, iteration_count, warm_up=warm_up)
        elif lr_policy == 'onecycle':
            lr = self.__schedule_one_cycle(optimizer, iteration_count)
        elif lr_policy == 'constant':
            lr = self.lr
        else:
            print(f'{lr_policy} is invalid lr policy')
            return None
        return lr

    def __set_lr(self, optimizer, lr):
        optimizer.__setattr__('lr', lr)

    def __set_momentum(self, optimizer, momentum):
        attr = ''
        optimizer_str = optimizer.__str__().lower()
        if optimizer_str.find('sgd') > -1:
            optimizer.__setattr__('momentum', momentum)
        elif optimizer_str.find('adam') > -1:
            optimizer.__setattr__('beta_1', momentum)

    def __warm_up_lr(self, iteration_count, warm_up):
        # return ((np.cos(((iteration_count * np.pi) / warm_up) + np.pi) + 1.0) * 0.5) * self.lr  # cosine warm up
        return self.lr * pow(iteration_count / float(warm_up), 4)

    def __schedule_step_decay(self, optimizer, iteration_count, warm_up=1000):
        if warm_up > 0 and iteration_count <= warm_up:
            lr = self.__warm_up_lr(iteration_count, warm_up)
        elif iteration_count >= int(self.iterations * 0.8):
            lr = self.lr * 0.1
        else:
            lr = self.lr
        self.__set_lr(optimizer, lr)
        return lr

    def __schedule_one_cycle(self, optimizer, iteration_count):
        warm_up = 0.3
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
        return lr

    def __schedule_cosine_warm_restart(self, optimizer, iteration_count, warm_up=1000):
        if warm_up > 0 and iteration_count <= warm_up:
            lr = self.__warm_up_lr(iteration_count, warm_up)
        else:
            if self.cycle_step % self.cycle_length == 0 and self.cycle_step != 0:
                self.cycle_step = 0
                self.cycle_length = int(self.cycle_length * self.cycle_weight)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(((1.0 / self.cycle_length) * np.pi * (self.cycle_step % self.cycle_length))))  # down and down
            self.cycle_step += 1
        self.__set_lr(optimizer, lr)
        return lr


def plot_lr(lr_policy):
    import tensorflow as tf
    from matplotlib import pyplot as plt
    lr = 0.001
    warm_up = 5000 
    iterations = warm_up + 37500
    optimizer = tf.keras.optimizers.SGD()
    lr_scheduler = LRScheduler(iterations=iterations, lr=lr)
    lrs = []
    for i in range(iterations):
        lr = lr_scheduler.update(optimizer=optimizer, iteration_count=i, warm_up=warm_up, lr_policy=lr_policy)
        lrs.append(lr)
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.legend(['lr'])
    plt.xlabel('iterations')
    plt.tight_layout(pad=0.5)
    plt.show()
    

if __name__ == '__main__':
    plot_lr('constant')
    plot_lr('step')
    plot_lr('onecycle')
    plot_lr('cosine')

