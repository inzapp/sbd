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
            validation_data_generator_flow=None):
        self.lr = lr
        self.burn_in = burn_in
        self.iteration_sum = 0
        self.batch_size = batch_size
        self.validation_data_generator_flow = validation_data_generator_flow
        self.max_map = 0.0
        self.max_f1 = 0.0
        self.max_hm = 0.0
        self.patience_count = 0
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
        elif self.patience_count >= 5:
            self.patience_count = 0
            self.lr *= 0.5
            if self.lr < 1e-4:
                self.lr = 1e-4
            print(f'[Learning rate reduce] lr : {self.lr:.6f}')
            lr = self.lr
        else:
            lr = self.lr
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.iteration_sum += 1
        if not curriculum_training and self.iteration_sum % 2000 == 0:
            self.save_model()

    @staticmethod
    def harmonic_mean(mean_ap, f1_score):
        return (2.0 * mean_ap * f1_score) / (mean_ap + f1_score + 1e-5)

    def is_better_than_before(self, mean_ap, f1_score):
        better_than_before = False
        if mean_ap > self.max_map:
            self.max_map = mean_ap
            better_than_before = True
        if f1_score > self.max_f1:
            self.max_f1 = f1_score
            better_than_before = True
        harmonic_mean = self.harmonic_mean(mean_ap, f1_score)
        if harmonic_mean > self.max_hm:
            self.max_hm = harmonic_mean
            better_than_before = True
        return better_than_before

    def save_model(self):
        print('\n')
        if self.validation_data_generator_flow is None:
            self.model.save(f'checkpoints/model_{self.iteration_sum}_iter.h5')
        elif self.iteration_sum >= 10000:
            self.model.save('model.h5', include_optimizer=False)
            mean_ap, f1_score = calc_mean_average_precision('model.h5', self.validation_data_generator_flow.image_paths)
            if self.is_better_than_before(mean_ap, f1_score):
                self.model.save(f'checkpoints/model_{self.iteration_sum}_iter_mAP_{mean_ap:.4f}_f1_{f1_score:.4f}.h5')
                self.patience_count = 0
            else:
                self.patience_count += 1

    def reset(self):
        self.iteration_sum = 0
        self.patience_count = 0
