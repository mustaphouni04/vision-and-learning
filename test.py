import cv2
from ultralytics import YOLO
import os
import random

# load YOLOv8 fine-tuned model
model = YOLO('models/best.pt')

# load class names
class_names = model.names

# set the directory path
dir_path = 'datasets/License_Plate_Detection.v1i.yolov9/valid/images'

# get a list of all images in the directory
image_list = [f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')]

# select 5 random images
random_images = random.sample(image_list, 35)

# iterate over the selected images
for image_name in random_images:
    image_path = os.path.join(dir_path, image_name)
    
    # read the image
    img = cv2.imread(image_path)
    
    # perform detection
    results = model(img)
    
    # loop through each detection in the results
    for result in results:
        boxes = result.boxes  # extract detected bounding boxes
        for box in boxes:
            # extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer values
            
            # get the class label and confidence score
            class_id = int(box.cls[0])
            label = class_names[class_id]
            conf = box.conf[0] * 100  # confidence in percentage
            
            # draw the bounding box and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {conf:.1f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # display the image with bounding boxes
    cv2.imshow('YOLO Detection', img)
    cv2.waitKey(0)  # press any key to move to the next image

# close all OpenCV windows
cv2.destroyAllWindows()

