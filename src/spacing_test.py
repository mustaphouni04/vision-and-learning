import cv2
import numpy as np
import imutils
from skimage import measure
from skimage.filters import threshold_local
from imutils import paths
from imutils import contours
from skimage.segmentation import clear_border
segmentation_spacing = 0.92
# Function to detect character candidates in a cropped license plate image
def detect_characters(plate):
    # Convert the plate image to HSV color space and extract the V channel
    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]

    # Apply adaptive thresholding to reveal the characters on the plate
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)

    # Resize the plate and threshold images for easier processing
    plate = imutils.resize(plate, width=400)
    thresh = imutils.resize(thresh, width=400)

    # Perform a connected components analysis on the thresholded image
    labels = measure.label(thresh, connectivity=2, background=0)

    # Mask to store the locations of the character candidates
    charCandidates = np.zeros(thresh.shape, dtype="uint8")

    # Loop over the unique components found by the connected components analysis
    for label in np.unique(labels):
        if label == 0:  # Ignore the background component
            continue

        # Create a mask for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255

        # Find contours in the label mask
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            # Grab the largest contour which corresponds to the component in the mask
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # Compute the aspect ratio, solidity, and height ratio for the component
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate.shape[0])

            # Apply the rules to determine if the contour is a likely character
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = 0.4 < heightRatio < 0.95

            # If the component passes all the tests, consider it a character candidate
            if keepAspectRatio and keepSolidity and keepHeight:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    # Remove any regions touching the border of the image
    charCandidates = measure.label(charCandidates, connectivity=2)
    charCandidates = cv2.bitwise_not(clear_border(cv2.bitwise_not(charCandidates)))
    charCandidates = charCandidates.astype(np.uint8)
    
    # Display the character candidates on the plate
    cv2.imshow("Thresholded Plate", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return thresh

# Example usage of the function with a cropped license plate image
plate_image_path = '../warped_test/warped_0918MCC_jpg.rf.0d52a87a35963f1606ba4d0886fe3f28.jpg'
plate = cv2.imread(plate_image_path)
thresh = detect_characters(plate)

'''3 Split characters'''
white = []  # Record the sum of white pixels in each column
black = []  # Record the sum of black pixels in each column
height = thresh.shape[0]
width = thresh.shape[1]

white_max = 0
black_max = 0


'''4 Cycle through the sum of black and white pixels for each column'''
for i in range(width):
    white_count = 0
    black_count = 0
    for j in range(height):
        if thresh[j][i] == 255:
            white_count += 1
        else:
            black_count += 1

    white.append(white_count)
    black.append(black_count)

white_max = max(white)
black_max = max(black)


'''5 Split the image, given the starting point of the character to be split'''
def find_end(start):
    end = start + 1
    for m in range(start + 1, width - 1):
        if(black[m] > segmentation_spacing * black_max):
            end = m
            break
    return end


n = 1
start = 1
end = 2

while n < width - 1:
    n += 1
    if(white[n] > (1 - segmentation_spacing) * white_max):
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            print(start, end)
            character = thresh[1:height, start:end]
            cv2.imwrite('img/{0}.png'.format(n), character)      
            cv2.imshow('character', character)
            cv2.imshow('image', thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()