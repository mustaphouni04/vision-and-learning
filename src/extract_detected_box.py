import cv2
from ultralytics import YOLO
import os

# load YOLOv9 fine-tuned model
model = YOLO('../models/best2.pt')

# load class names
class_names = model.names

# set the directory paths
valid_dir_path = '../datasets/License_Plate_Detection.v2i.yolov9/valid/images'
train_dir_path = '../datasets/License_Plate_Detection.v2i.yolov9/train/images'
test_dir_path = '../datasets/License_Plate_Detection.v2i.yolov9/test/images'

# create directories to store cropped images
valid_cropped_dir_path = '../datasets/License_Plate_Detection.v2i.yolov9/valid/cropped'
train_cropped_dir_path = '../datasets/License_Plate_Detection.v2i.yolov9/train/cropped'
test_cropped_dir_path = '../datasets/License_Plate_Detection.v2i.yolov9/test/cropped'

# create directories if they don't exist
os.makedirs(valid_cropped_dir_path, exist_ok=True)
os.makedirs(train_cropped_dir_path, exist_ok=True)
os.makedirs(test_cropped_dir_path, exist_ok=True)

# get a list of all images in the directories
valid_image_list = [f for f in os.listdir(valid_dir_path) if f.endswith('.jpg') or f.endswith('.png')]
train_image_list = [f for f in os.listdir(train_dir_path) if f.endswith('.jpg') or f.endswith('.png')]
test_image_list = [f for f in os.listdir(test_dir_path) if f.endswith('.jpg') or f.endswith('.png')]

# iterate over all images in the directories
for image_name in valid_image_list + train_image_list + test_image_list:
    if image_name in valid_image_list:
        dir_path = valid_dir_path
        cropped_dir_path = valid_cropped_dir_path
    elif image_name in train_image_list:
        dir_path = train_dir_path
        cropped_dir_path = train_cropped_dir_path
    else:
        dir_path = test_dir_path
        cropped_dir_path = test_cropped_dir_path
    
    image_path = os.path.join(dir_path, image_name)
    
    # read the image
    img = cv2.imread(image_path)
    
    # perform detection
    results = model(img)
        
    # save the cropped boundaries of the detected plates
    max_conf = 0
    max_conf_box = None
    for result in results:
        boxes = result.boxes  # extract detected bounding boxes
        for box in boxes:
            # extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to integer values
            
            # get the class label and confidence score
            class_id = int(box.cls[0])
            label = class_names[class_id]
            conf = box.conf[0] * 100  # confidence in percentage
            
            if conf > max_conf:
                max_conf = conf
                max_conf_box = (x1, y1, x2, y2)
    
    if max_conf_box is not None:
        x1, y1, x2, y2 = max_conf_box
        cv2.imwrite(f'{cropped_dir_path}/{image_name}_{label}_{max_conf:.1f}.jpg', img[y1:y2, x1:x2])

print("DONE")
