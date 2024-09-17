import cv2
import os

folder_path = '../images/Test/Frontal/Frontal'

image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
images = []

# open each image file and display it
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path,0)
    images.append((image_file,image))

# Load the template image
template = cv2.imread('template.png', 0)

# Load the image that contains the license plate
img = images[5][1]
width = 1600
height = 900
coord = (width, height)

img = cv2.resize(img, coord, interpolation=cv2.INTER_AREA)
# Apply thresholding to enhance contrast
template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform template matching
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Find the maximum value in the result matrix
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
cv2.rectangle(img, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 255, 0), 2)

# Display the result
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()