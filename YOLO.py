import cv2
from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')
dataset_config = "datasets/License_Plate_Detection.v1i.yolov9/data.yaml"

results = model.train(data=dataset_config, epochs=5)