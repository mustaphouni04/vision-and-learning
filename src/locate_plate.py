# import the necessary packages
from imutils import paths
import imutils
import cv2
from skimage.segmentation import clear_border
import numpy as np

# a hyperparameter that changes the aspect ratio of the rectangles
min_aspect_ratio = 3
max_aspect_ratio = 6

# here's where all morph operations are done to extract the ROI
def locate_license_plate_candidates(gray, keep=5):
    # perform a blackhat morphological operation that will allow
    # us to reveal dark regions (i.e., text) on light backgrounds
    # (i.e., the license plate itself)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    # next, find regions in the image that are light
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # compute the Scharr gradient representation of the blackhat
    # image in the x-direction and then scale the result back to
    # the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
        dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    # blur the gradient representation, applying a closing
    # operation, and threshold the image using Otsu's method
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform a series of erosions and dilations to clean up the
    # thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # take the bitwise AND between the threshold result and the
    # light regions of the image
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
    # initialize the license plate contour and ROI
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
            # (which typically, not but always, indicates noise)
            if clearBorder:
                roi = clear_border(roi)
            # display any debugging information and then break
            # from the loop early since we have found the license
            # plate region
            break
    # return a 2-tuple of the license plate ROI and the contour
    # associated with it
    return (roi, lpCnt)

def find_and_ocr(image, clearBorder=False):
    # convert the input image to grayscale, locate all candidate
    # license plate regions in the image, and then process the
    # candidates, leaving us with the *actual* license plate
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = locate_license_plate_candidates(gray)
    (lp, lpCnt) = locate_license_plate(gray, candidates,
        clearBorder=clearBorder)
    # return the detected region of the license plate
    return lp

# grab all image paths in the input directory
imagePaths = sorted(list(paths.list_images("../../images/Test")))

# loop over all image paths in the input directory
for imagePath in imagePaths:
    # load the input image from disk and resize it
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    # apply automatic license plate recognition
    lp = find_and_ocr(image, clearBorder=False)
    # only continue if the license plate was successfully detected
    if lp is not None:
        # display the detected region of the license plate
        cv2.imshow("Detected License Plate", lp)
        cv2.waitKey(0)