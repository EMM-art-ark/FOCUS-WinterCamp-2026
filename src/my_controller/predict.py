from ultralytics import YOLO
predictor = YOLO(model='runs/detect/train11/weights/best.pt',task='detect')
result = predictor(source='1.jpg',save=True,cfg=0.05)