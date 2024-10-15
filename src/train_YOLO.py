import cv2
from ultralytics import YOLO

model = YOLO('../models/yolov8n-seg.pt')
dataset_config = "../datasets/License_Plate_Segmentation.v1i.yolov8/data.yaml"

results = model.train(data=dataset_config, epochs=20)