# libs
import cv2
import numpy as np

# downscale img
img = cv2.imread('../images/Test/Frontal/Frontal/1062FNT.jpg', cv2.IMREAD_GRAYSCALE)
width = 1920
height = 1080
coord = (width, height)

re_img = cv2.resize(img, coord, interpolation=cv2.INTER_AREA)

# binary inverted threshold due to license being white and otsu for optimal threshold value
_, thresh = cv2.threshold(re_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# eroding and dilating the thresholded img
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
eroded = cv2.erode(thresh, kernel, iterations=5)
dilated = cv2.dilate(eroded, kernel, iterations=1)
edge = cv2.Canny(dilated, 100, 200)

# find contours in the img / only retrieve outermost contours + store only essential contour points
contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# out is a list of the contours

# filter the contours found in the img
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour) # create a bounding rectangle for the contour (x,y) coords are top left side of rectagle
    aspect_ratio = float(w)/h
    area = cv2.contourArea(contour)
    if aspect_ratio > 2 and area > 1000: # heuristic to find license plate
        # draw a bounding rectangle around the contour
        cv2.rectangle(re_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # extract the region of interest
        roi = re_img[y:y+h, x:x+w]
        break

while True:
    cv2.imshow("Window", re_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

