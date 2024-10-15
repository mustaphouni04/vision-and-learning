import cv2
import os
import numpy as np

# Set folder paths
input_folder_path = "../warped_test"
output_folder_path = "segmented_images"

# Create the output directory if it does not exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Loop through all files in the input folder
for file_name in os.listdir(input_folder_path):
    # Create the full path to the image file
    image_path = os.path.join(input_folder_path, file_name)
    print(f"Processing {image_path}")

    # Load the image in grayscale
    gray = cv2.imread(image_path, 0)

    # Resize the image to make the characters larger
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray, 3)

    # Thresholding to create a binary image
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Create a rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply dilation 
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)

    # Find contours
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the image for drawing bounding boxes
    im2 = gray.copy()

    # Loop through contours and find letters in the license plate
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = im2.shape
        
        # Criteria for filtering contours
        if height / float(h) > 6: continue
        ratio = h / float(w)
        if ratio < 1.5: continue
        area = h * w
        if area < 100: continue
        
        # Draw the rectangle around detected characters
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the image with bounding boxes
    marked_image_path = os.path.join(output_folder_path, f"{file_name}_marked.png")
    cv2.imwrite(marked_image_path, im2)  # Save the marked image

    print(f"Saved marked image as {marked_image_path}\n")

print("Processing complete.")
