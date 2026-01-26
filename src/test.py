from ultralytics import YOLO
test = YOLO(model='runs/detect/train8/weights/best.pt', task='detect')
result = test(source='xr.jpg', save=True)