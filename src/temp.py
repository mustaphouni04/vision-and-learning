import cv2
import os

folder_path = '../../images/Test/Frontal/Frontal'

image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
images = []

# open each image file and display it
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path,0)
    images.append((image_file,image))

# load the template image
template = cv2.imread('../template.png', 0)

# apply thresholding to enhance contrast
template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# set the threshold for the maximum value in the result matrix
threshold = 0.225

# process each image
for i, (image_file, img) in enumerate(images):
    # Resize the image
    width = 1600
    height = 900
    coord = (width, height)
    img = cv2.resize(img, coord, interpolation=cv2.INTER_AREA)

    # apply thresholding to enhance contrast
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # perform template matching
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # find the maximum value in the result matrix
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # check if the maximum value is above the threshold
    if max_val >= threshold:
        # draw a rectangle around the matched region
        cv2.rectangle(img, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 255, 0), 2)

        # display the result
        cv2.imshow('Result - ' + image_file, img)
    else:
        print(f"No match found for {image_file}")

# wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()