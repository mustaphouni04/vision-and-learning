import pytesseract
import cv2
import os
import numpy as np

# set folder path
folder_path = "../warped_test"

# loop through all files in the folder
for file_name in os.listdir(folder_path):
    # create the full path to the image file
    image_path = os.path.join(folder_path, file_name)
    print(image_path)
    # load the image in grayscale
    gray = cv2.imread(image_path, 0)

    # resize the image to make the characters larger
    gray = cv2.resize(gray, None, fx=5, fy=5, interpolation= cv2.INTER_LINEAR)

    # apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # apply median blur for further noise reduction
    gray = cv2.medianBlur(blur, 3)

    # thresholding to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # create a rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilation = cv2.dilate(thresh, rect_kern, iterations=2)

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours from left to right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # create a copy of the image to draw rectangles
    im2 = gray.copy()

    plate_num = ""
    # loop through contours and find letters in the license plate
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = im2.shape
        ratio = h / float(w)

        area = h * w
        
        # draw the rectangle around each detected character
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = im2[y - 5:y + h + 5, x - 5:x + w + 5]
        if roi.size == 0:
            print(f"Warning: Empty ROI for contour in {image_path}. Skipping this contour.")
            continue
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)

        # recognize text using Tesseract OCR
        text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        print(text)
        plate_num += text

    # print the recognized plate number
    print(f"Recognized Plate Number for {file_name}: {plate_num}\n")

    # display the image with character segmentation
    cv2.imshow(f"Character's Segmented - {file_name}", im2)
    cv2.waitKey(0)

# close all OpenCV windows
cv2.destroyAllWindows()
