# Import the necessary packages
import cv2
import os
import numpy as np
from skimage.segmentation import clear_border
import imutils
from imutils import paths
import pytesseract
from difflib import SequenceMatcher
import json

# Hyperparameters
min_aspect_ratio = 5
max_aspect_ratio = 8
output_folder = "extracted_plates"  # Folder to save detected plates
output_with_text_folder = "output_with_text"  # Folder to save images with text

# Create the output folders if they do not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_with_text_folder):
    os.makedirs(output_with_text_folder)

# Function to find license plate candidates using contour detection
def locate_license_plate_candidates(gray, keep=5):
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
    return cnts

# Function to locate the license plate region from the image
def locate_license_plate(gray, candidates, clearBorder=False):
    lpCnt = None
    roi = None
    for c in candidates:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if min_aspect_ratio <= ar <= max_aspect_ratio:
            lpCnt = c
            licensePlate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            if clearBorder:
                roi = clear_border(roi)
            break
    return (roi, lpCnt)

# Function to build Tesseract options for OCR
def build_tesseract_options(psm=7):
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(psm)
    return options

# Function to extract the license plate and recognize its text using Tesseract
def find_plate(image, clearBorder=False, psm=7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = locate_license_plate_candidates(gray)
    (lp, lpCnt) = locate_license_plate(gray, candidates, clearBorder=clearBorder)
    lpText = None
    if lp is not None:
        options = build_tesseract_options(psm=psm)
        lpText = pytesseract.image_to_string(lp, config=options)
    return (lp, lpText)

# Function to clean up the recognized text
def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()
    
def calculate_levenshtein_distance(a, b):
    return int((1 - SequenceMatcher(None, a, b).ratio()) * max(len(a), len(b)))
evaluation_metrics = []

# Grab all image paths in the input directory
imagePaths = sorted(list(paths.list_images("../original_images")))

# Loop over all image paths in the input directory
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=575)
    lp, lpText = find_plate(image, clearBorder=True)

    # Extract the ground truth text from the image filename
    ground_truth = os.path.splitext(os.path.basename(imagePath))[0]
    
    if lp is not None:
        # Clean up the recognized text
        cleaned_text = cleanup_text(lpText)
        
        # Calculate Levenshtein distance
        levenshtein_distance = calculate_levenshtein_distance(cleaned_text, ground_truth)
        
        # Save metrics for this sample
        evaluation_metrics.append({
            "image": os.path.basename(imagePath),
            "ground_truth": ground_truth,
            "predicted_text": cleaned_text,
            "levenshtein_distance": levenshtein_distance
        })
        
        # Draw the recognized text on the image
        cv2.putText(image, cleaned_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save the image with recognized text in a separate folder
        output_with_text_path = os.path.join(output_with_text_folder, os.path.basename(imagePath))
        cv2.imwrite(output_with_text_path, image)

        # Save the detected license plate image
        cv2.imwrite(os.path.join(output_folder, os.path.basename(imagePath)), lp)

        print(f"Recognized Plate Text for {os.path.basename(imagePath)}: {cleaned_text}")

# Save the evaluation metrics to a JSON file for later analysis
with open("evaluation_metrics.json", "w") as metrics_file:
    json.dump(evaluation_metrics, metrics_file, indent=4)

print("Evaluation metrics saved to evaluation_metrics.json")

# Initialize variables for global metrics
total_levenshtein_distance = 0
total_characters = 0
correct_predictions = 0
total_predictions = len(evaluation_metrics)

# Calculate global metrics
for metric in evaluation_metrics:
    total_levenshtein_distance += metric["levenshtein_distance"]
    total_characters += len(metric["ground_truth"])
    if metric["ground_truth"] == metric["predicted_text"]:
        correct_predictions += 1

# Calculate Average Levenshtein Distance, Character Error Rate (CER), and Word Accuracy Rate (WAR)
average_levenshtein_distance = total_levenshtein_distance / total_predictions
character_error_rate = total_levenshtein_distance / total_characters
word_accuracy_rate = (correct_predictions / total_predictions) * 100

# Print the global metrics
print(f"Average Levenshtein Distance: {average_levenshtein_distance:.2f}")
print(f"Character Error Rate (CER): {character_error_rate:.2%}")
print(f"Word Accuracy Rate (WAR): {word_accuracy_rate:.2f}%")


