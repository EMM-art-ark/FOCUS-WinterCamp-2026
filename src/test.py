from ultralytics import YOLO
test = YOLO(model='runs/detect/train8/weights/best.pt', task='detect')

result = test(source='test1.jpg', save=True)
