from yolo import Yolo

if __name__ == '__main__':
    model = Yolo()
    model.fit(
        train_image_path=r'C:\inz\train_data\lp_detection_yolo',
        input_shape=(368, 640, 1),
        batch_size=2,
        lr=1e-2,
        epochs=300,
        validation_split=0.2)
