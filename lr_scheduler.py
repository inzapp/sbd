import os

import numpy as np
import tensorflow as tf

from mAP_calculator import calc_mean_average_precision


class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(
            self,
            lr=0.001,
            burn_in=1000,
            batch_size=32,
            iterations=100000,
            validation_data_generator_flow=None):
        self.lr = lr
        self.burn_in = burn_in
        self.iteration_sum = 0
        self.batch_size = batch_size
        self.iterations = iterations
        self.validation_data_generator_flow = validation_data_generator_flow
        super().__init__()

    def on_train_begin(self, logs=None):
        if not (os.path.exists('checkpoints') and os.path.exists('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

    def on_train_batch_begin(self, batch, logs=None):
        self.update(self.model)

    def update(self, model, curriculum_training=False):
        self.model = model
        if self.iteration_sum < self.burn_in:
            lr = self.lr * self.batch_size / self.burn_in
        elif self.iteration_sum < self.burn_in * 2:
            warmup_lr = self.lr * self.batch_size / self.burn_in
            lr = warmup_lr + 0.5 * (self.lr - warmup_lr) * (1.0 + np.cos(np.pi * (self.iteration_sum - self.burn_in) / self.burn_in + np.pi))
        else:
            lr = self.lr
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.iteration_sum += 1
        if not curriculum_training and self.iteration_sum % 2000 == 0:
            self.save_model()

    def save_model(self):
        print('\n')
        if self.validation_data_generator_flow is None:
            self.model.save(f'checkpoints/model_{self.iteration_sum}_iter.h5')
        else:
            self.model.save('model.h5', include_optimizer=False)
            mean_ap, f1_score = calc_mean_average_precision('model.h5', self.validation_data_generator_flow.image_paths)
            self.model.save(f'checkpoints/model_{self.iteration_sum}_iter_mAP_{mean_ap:.4f}_f1_{f1_score:.4f}.h5')

    def reset(self):
        self.iteration_sum = 0
