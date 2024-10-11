# import the necessary packages
from imutils import paths
import imutils
import cv2
from skimage.segmentation import clear_border # assists in cleaning up the borders of images
import numpy as np
import matplotlib.pyplot as plt

# a hyperparameter that changes the aspect ratio of the rectangles
min_aspect_ratio = 3 # the minimum aspect ratio used to detect and filter rectangular license plates
max_aspect_ratio = 6 # the maximum aspect ratio of the license plate rectangle

# find the license plate candidate contours
def locate_license_plate_candidates(gray, keep=5): # only return up to this many sorted license plate candidate contours
    # blackhat to reveal black against white
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    # find light regions in the image (close all black in the license plate to reveal a mask)
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # compute Sobel in X, emphasizing edges in characters and removing unnecessary Y edges
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
        dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    # blur edges and close (black dots inside white space) + threshold the img
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform a series of erosions and dilations to clean up the thresh img
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # take the bitwise AND between the threshold result and the light regions of the image
    # all vals that are 1 in the mask will be 1 in thresh, otherwise 0
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    # find contours in the thresholded image and sort them by
    # their size in descending order, keeping only the largest
    # ones
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
    # return the list of contours
    return cnts

def locate_license_plate(gray, candidates,
        clearBorder=False):
    lpCnt = None
    roi = None
    # loop over the license plate candidate contours
    for c in candidates:
        # compute the bounding box of the contour and then use
        # the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # check to see if the aspect ratio is rectangular
        if ar >= min_aspect_ratio and ar <= max_aspect_ratio:
            # store the license plate contour and extract the
            # license plate from the grayscale image and then
            # threshold it
            lpCnt = c
            licensePlate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # check to see if we should clear any foreground
            # pixels touching the border of the image
            if clearBorder:
                roi = clear_border(roi)
            break
            
    return (roi, lpCnt)

def find_plate(image, clearBorder=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = locate_license_plate_candidates(gray)
    (lp, lpCnt) = locate_license_plate(gray, candidates,
        clearBorder=clearBorder)
    # return the detected region of the license plate
    return lp

# grab all image paths in the input directory
imagePaths = sorted(list(paths.list_images("../original_images")))

# loop over all image paths in the input directory
for imagePath in imagePaths:
    # load the input image from disk and resize it
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=500)
    # apply automatic license plate recognition
    lp = find_plate(image, clearBorder=True)
    # only continue if the license plate was successfully detected
    if lp is not None:
        # display the detected region of the license plate
        plt.imshow(lp, cmap='gray')
        plt.title(f"{imagePath}")
        plt.axis('off')
        plt.show()