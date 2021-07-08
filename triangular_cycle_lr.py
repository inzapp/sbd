import tensorflow as tf


class TriangularCycleLR(tf.keras.callbacks.Callback):

    def __init__(
            self,
            max_lr=0.1,
            min_lr=1e-5,
            cycle_step=2000,
            batch_size=None,
            train_data_generator=None,
            validation_data_generator=None):
        self.batch_count = 0
        self.batch_sum = 0
        self.lr = min_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_step = cycle_step
        self.lr_offset = (max_lr - min_lr) / float(cycle_step / 2.0)
        self.batch_size = batch_size
        self.train_data_generator = train_data_generator
        self.validation_data_generator = validation_data_generator
        self.increasing = True
        super().__init__()

    def on_batch_end(self, epoch, logs=None):
        self.batch_count += 1
        self.batch_sum += 1
        if self.batch_count == self.cycle_step:
            self.save_model(with_loss=True)

        if self.batch_count == int(self.cycle_step / 2 + 1):
            self.increasing = False
        elif self.batch_count == self.cycle_step + 1:
            self.increasing = True
            self.batch_count = 1
            self.max_lr *= 0.9
            self.lr_offset = (self.max_lr - self.min_lr) / float((self.cycle_step - 2) / 2.0)

        if self.increasing:
            self.increase_lr()
        else:
            self.decrease_lr()

    def decrease_lr(self):
        self.lr -= self.lr_offset
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def increase_lr(self):
        self.lr += self.lr_offset
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def save_model(self, with_loss=False):
        if with_loss:
            loss = self.model.evaluate(x=self.train_data_generator.flow(), batch_size=self.batch_size, return_dict=True)['loss']
            val_loss = self.model.evaluate(x=self.validation_data_generator.flow(), batch_size=self.batch_size, return_dict=True)['loss']
            print(f'{self.batch_sum} batch => loss: {loss:.4f}, val_loss: {val_loss:.4f}\n')
            self.model.save(f'checkpoints/model_{self.batch_sum}_batch_loss_{loss:.4f}_val_loss_{val_loss:.4f}.h5')
        else:
            self.model.save(f'checkpoints/model_{self.batch_sum}_batch.h5')
