import os

import numpy as np
import tensorflow as tf

from mAP_calculator import calc_mean_average_precision


class CosineLRDecay(tf.keras.callbacks.Callback):
    def __init__(
            self,
            max_lr=0.1,
            min_lr=1e-4,
            batch_size=32,
            cycle_length=1000,
            train_data_generator_flow=None,
            validation_data_generator_flow=None):
        self.iteration_sum = 0
        self.iteration_count = 0
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.train_data_generator_flow = train_data_generator_flow
        self.validation_data_generator_flow = validation_data_generator_flow
        super().__init__()

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
        if not (os.path.exists('checkpoints') and os.path.exists('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

    def on_train_batch_begin(self, batch, logs=None):
        self.update(self.model)

    def update(self, model):
        self.model = model
        if self.iteration_sum < self.cycle_length:
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(np.pi * self.iteration_count / self.cycle_length + np.pi))  # warm up
        else:
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(np.pi * self.iteration_count / self.cycle_length))  # decay
        print(f'{self.iteration_count} => {lr:.4f}')
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.iteration_count += 1
        if self.iteration_count == self.cycle_length:
            self.iteration_count = 0

        self.iteration_sum += 1
        if self.iteration_sum % self.cycle_length == 0:
            self.save_model()

    def save_model(self):
        if self.train_data_generator_flow is None or self.validation_data_generator_flow is None:
            self.model.save(f'checkpoints/model_{self.iteration_sum}_batch.h5')
        else:
            self.model.save('model.h5')
            print(f'[{self.iteration_sum} iterations]')
            mean_ap, f1_score = calc_mean_average_precision('model.h5', self.validation_data_generator_flow.image_paths)
            self.model.save(f'checkpoints/model_{self.iteration_sum}_iter_mAP_{mean_ap:.4f}_f1_{f1_score:.4f}.h5')
