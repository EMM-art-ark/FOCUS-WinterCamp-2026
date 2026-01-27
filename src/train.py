from ultralytics import YOLO
model = YOLO(model='yolo26n.pt')
model.train(data='focus.yaml',workers=0,epochs=500,batch=16,device=[-1,-1])