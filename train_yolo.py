from ultralytics import YOLO
model=YOLO('yolov8.yaml')
results=model.train(data= r"C:\Users\DELL\Downloads\UA-DETRAC-DATASET-10K.v2-2024-11-14-3-48pm.yolov11\data.yaml",epoch=50,device=0,pretrained=True)
