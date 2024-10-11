import cv2
import numpy as np
from ultralytics import YOLO
import os

# load YOLOv8 fine-tuned model
model = YOLO('../models/best-seg.pt')

# load class names
class_names = model.names

# set the directories for train, test, and valid image folders
data_folders = {
    'train': '../datasets/License_Plate_Segmentation.v1i.yolov8/train/images',
    'test': '../datasets/License_Plate_Segmentation.v1i.yolov8/test/images',
    'valid': '../datasets/License_Plate_Segmentation.v1i.yolov8/valid/images'
}

# output directories for warped images
output_folders = {
    'train': '../warped_train',
    'test': '../warped_test',
    'valid': '../warped_valid'
}

# ensure output directories exist
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# function to order points for perspective transform
def order_points(pts):
    # initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")
    
    # the top-left point will have the smallest sum, the bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # the top-right point will have the smallest difference, the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# process each dataset (train, test, valid)
for dataset_type, dir_path in data_folders.items():
    # get a list of all images in the directory
    image_list = [f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')]

    # iterate over the images
    for image_name in image_list:
        image_path = os.path.join(dir_path, image_name)
        
        # read the image
        img = cv2.imread(image_path)
        
        # perform detection
        results = model(img)
        
        # loop through each detection in the results
        for result in results:
            masks = result.masks  # extract detected segmentation masks
            if masks is not None:  # check if masks are present
                for mask in masks:
                    # retrieve the mask coordinates (polygon points)
                    polygon_points = mask.xy  # list of (x, y) coordinates

                    # convert the polygon points to a NumPy array
                    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 2))

                    # get the 4 corner points of the license plate
                    rect = order_points(pts)
                    
                    # determine the width and height of the new image
                    (tl, tr, br, bl) = rect
                    widthA = np.linalg.norm(br - bl)
                    widthB = np.linalg.norm(tr - tl)
                    maxWidth = max(int(widthA), int(widthB))

                    heightA = np.linalg.norm(tr - br)
                    heightB = np.linalg.norm(tl - bl)
                    maxHeight = max(int(heightA), int(heightB))

                    # set up the destination points for the perspective transform
                    dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")

                    # compute the perspective transform matrix
                    M = cv2.getPerspectiveTransform(rect, dst)

                    # warp the image to get the license plate horizontal
                    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

                    # save the warped license plate image to the appropriate folder
                    output_filename = os.path.join(output_folders[dataset_type], f"warped_{image_name}")
                    cv2.imwrite(output_filename, warped)

                    # draw the polygon on the original image for visualization
                    cv2.polylines(img, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)

        # cv2.imshow('YOLO Segmentation with Polygon', img)
        # cv2.waitKey(0)  

#cv2.destroyAllWindows()



