from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n-cls.pt')

    model.train(
        data='datasets/trash_data',
        epochs=30,
        imgsz=640,
        batch=4,
        device="cpu", #gpu가 없으면 "cpu" gpu있으면 0
	workers=2
    )
