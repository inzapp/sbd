from yolo import Yolo

if __name__ == '__main__':
    model = Yolo()
    model.fit(
        train_image_path=r'C:\inz\train_data\lp_character_detection',
        input_shape=(96, 192, 1),
        batch_size=2,
        lr=1e-4,
        epochs=1000,
        validation_split=0.2)
    model.evaluate()
