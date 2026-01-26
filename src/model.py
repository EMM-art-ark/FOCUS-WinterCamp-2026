from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.train(data='ultralytics/cfg/datasets/coco128.yaml', workers=0, epochs=100, batch=16, device=-1)