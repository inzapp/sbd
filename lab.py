import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def main():
    # model = tf.keras.models.load_model('checkpoints/model_epoch_1_loss_0.0186_val_loss_0.0157.h5')
    model = tf.keras.models.load_model('model.h5')
    # model = tf.keras.models.load_model('lp_type_model.h5')
    previous_channel = -1
    total_convolution_count = 0
    for layer in model.layers:
        if str(layer).lower().find('conv') == -1:
            continue
        if previous_channel == -1:
            previous_channel = layer.input_shape[3]
        shape = layer.output_shape
        if type(shape) is list:
            shape = shape[0]
        h, w, c = shape[1:]
        d = tf.keras.utils.serialize_keras_object(layer)
        kernel_size = d['config']['kernel_size']
        strides = d['config']['strides']
        cur_convolution_count = h * w * c * previous_channel
        cur_convolution_count = cur_convolution_count * kernel_size[0] * kernel_size[1]
        cur_convolution_count = cur_convolution_count / (strides[0] * strides[1])
        total_convolution_count += cur_convolution_count
        previous_channel = c
    print(total_convolution_count)


def p(v):
    try:
        len(v)
        for c in v:
            print(f'{c:.4f} ', end='')
        print()
    except TypeError:
        print(f'{v:.4f}')


class SumOverLogError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mae = tf.math.abs(tf.math.subtract(y_pred, y_true))
        sub = tf.math.subtract(tf.constant(1.0 + 1e-7), mae)
        log = -tf.math.log(sub)
        return tf.keras.backend.mean(tf.keras.backend.sum(log, axis=-1))


class FalsePositiveWeightedError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        fp = tf.keras.backend.clip(-(y_true - y_pred), 0.0, 1.0)
        fp = -tf.math.log(1.0 + 1e-7 - fp)
        fp = tf.keras.backend.mean(tf.keras.backend.sum(fp, axis=-1))
        loss = tf.math.abs(y_pred - y_true)
        loss = -tf.math.log(1.0 + 1e-7 - loss)
        loss = tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
        return loss + fp


class MeanAbsoluteLogError(tf.keras.losses.Loss):
    """
    False positive weighted loss function.
    f(x) = -log(1 - MAE(x))
    Usage:
     model.compile(loss=[MeanAbsoluteLogError()], optimizer="sgd")
    """

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = tf.math.abs(y_pred - y_true)
        loss = -tf.math.log(1.0 + 1e-7 - loss)
        loss = tf.keras.backend.mean(tf.keras.backend.mean(loss, axis=-1))
        return loss


class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[BinaryFocalLoss(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer="adam")
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
        super().__init__()

    def call(self, y_true, y_pred):
        from keras import backend as K
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * self.alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), self.gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))


def test_interpolation():
    from time import time
    from time import sleep
    x = cv2.imread(r'C:\inz\fhd.jpg', cv2.IMREAD_GRAYSCALE)

    st = time()
    for i in range(100):
        linear = cv2.resize(x, (640, 368), interpolation=cv2.INTER_LINEAR)
    et = time()
    print(f'linear : {(et - st):.64f}')
    cv2.imshow('x', linear)
    cv2.waitKey(0)

    sleep(0.1)

    st = time()
    for i in range(100):
        area = cv2.resize(x, (640, 368), interpolation=cv2.INTER_AREA)
    et = time()
    print(f'area   : {(et - st):.64f}')
    cv2.imshow('x', area)
    cv2.waitKey(0)

    sleep(0.1)

    st = time()
    for i in range(100):
        nearest = cv2.resize(x, (640, 368), interpolation=cv2.INTER_NEAREST)
    et = time()
    print(f'nearest: {(et - st):.64f}')
    cv2.imshow('x', nearest)
    cv2.waitKey(0)
    pass


def test_loss():
    # y_true = [
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ]
    # y_pred = [
    #     [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ]
    # y_pred = [
    #     [0.5, 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0.5, 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
    #     [0.5, 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ]

    # y_true = [
    #     [0.25, 0.50, 0.25],
    #     [0.50, 1.00, 0.50],
    #     [0.25, 0.50, 0.25]
    # ]
    # y_pred = [
    #     [0.10, 0.40, 0.10],
    #     [0.40, 1.00, 0.40],
    #     [0.10, 0.40, 0.10]
    # ]

    y_true = [[0.0, 0.0, 1.0, 0.0, 0.0]]
    y_pred = [[0.8, 0.35, 0.75, 0.05, 0.03]]

    # y_true = [[0.0]]
    # y_pred = [[1.0]]

    # y_true = [
    #     [
    #         [
    #             [1.0, 0.0],
    #             [0.0, 0.0]
    #         ],
    #         [
    #             [1.0, 0.0],
    #             [0.0, 0.0]
    #         ],
    #     ],
    # ]
    #
    # y_pred = [
    #     [
    #         [
    #             [0.0, 0.0],
    #             [0.0, 0.0]
    #         ],
    #         [
    #             [1.0, 0.0],
    #             [0.0, 0.0]
    #         ],
    #     ],
    # ]

    # y_true = [[0. for _ in range(1000)]]
    # y_true[0][50] = 1.0
    # y_pred = [[0. for _ in range(1000)]]
    # y_pred[0][50] = 0.5

    p(BinaryFocalLoss()(y_true, y_pred).numpy())
    p(MeanAbsoluteLogError()(y_true, y_pred).numpy())
    p(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())


def bounding_box_test():
    x = np.asarray([
        [0, 127, 50],
        [127, 255, 0],
        [0, 50, 0]]
    ).astype('uint8')
    x = cv2.resize(x, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('x', x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def interpolation_test():
    x = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.0, 0.4, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.0, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    )
    x = np.clip(x, 0.7, 1.0)
    cv2.normalize(x, x, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    x = (x * 255.0).astype('uint8')
    x = cv2.resize(x, (500, 500), interpolation=cv2.INTER_LINEAR)

    for i in range(100):
        x_copy = x.copy()
        threshold = float(i / 100.0)
        _, y = cv2.threshold(x, int(255.0 * threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            for contour in contours:
                x1, y1, w, h = cv2.boundingRect(contour)
                cv2.rectangle(x_copy, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), thickness=2)
        print(threshold)
        cv2.imshow('x', x_copy)
        cv2.imshow('y', y)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_1_box_label():
    from glob import glob
    from tqdm import tqdm
    import yolo
    with open(f'{yolo.train_image_path}/classes.txt', 'rt') as classes_file:
        yolo.class_names = [s.replace('\n', '') for s in classes_file.readlines()]
    img_paths = glob(rf'{yolo.train_image_path}\*\*.jpg')
    for cur_img_path in tqdm(img_paths):
        x = cv2.imread(cur_img_path, yolo.img_type)
        x = yolo.resize(x, (yolo.input_shape[1], yolo.input_shape[0]))
        with open(rf'{cur_img_path[:-4]}.txt', mode='rt') as f:
            lines = f.readlines()
        y = np.zeros((yolo.input_shape[0], yolo.input_shape[1]), dtype=np.uint8)
        for line in lines:
            class_index, cx, cy, w, h = list(map(float, line.split(' ')))
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1, y1, x2, y2 = int(x1 * x.shape[1]), int(y1 * x.shape[0]), int(x2 * x.shape[1]), int(y2 * x.shape[0])
            cv2.rectangle(y, (x1, y1), (x2, y2), (255, 255, 255), -1)

        contours, _ = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        converted_label_str = ''
        for contour in contours:
            x_raw, y_raw, width_raw, height_raw = cv2.boundingRect(contour)
            x_min_f = x_raw / float(yolo.input_shape[1])
            y_min_f = y_raw / float(yolo.input_shape[0])
            width_f = width_raw / float(yolo.input_shape[1])
            height_f = height_raw / float(yolo.input_shape[0])
            cx_f = x_min_f + width_f / 2.0
            cy_f = y_min_f + height_f / 2.0
            converted_label_str += f'0 {cx_f:.6f} {cy_f:.6f} {width_f:.6f} {height_f:.6f}\n'
        with open(rf'{cur_img_path[:-4]}.txt', mode='wt') as f:
            f.write(converted_label_str)


if __name__ == '__main__':
    convert_1_box_label()
    # test_loss()
    # bounding_box_test()
    # test_interpolation()
