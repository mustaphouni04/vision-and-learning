import cv2
import joblib
import matplotlib.pyplot as plt
import os
import matplotlib
import csv
import numpy as np
from sklearn.metrics import accuracy_score
import Levenshtein

matplotlib.use('Agg')  # Use Agg backend for saving images

# Ordinary license plate value is 0.95, new energy license plate is changed to 0.9
segmentation_spacing = 0.65

# Create the output directories if they do not exist
chars_output_dir = "../chars"
grids_output_dir = "grids"
results_file = "ocr_results.csv"

if not os.path.exists(chars_output_dir):
    os.makedirs(chars_output_dir)

if not os.path.exists(grids_output_dir):
    os.makedirs(grids_output_dir)

# Load the pre-trained character recognition model
loaded_model = joblib.load("../models/svm_character_recognition_model.pkl")

# Function to display a grid of character images using matplotlib
def display_characters_grid(characters, labels, save_path=None):
    num_characters = len(characters)

    if num_characters == 0:
        print("No characters to display.")
        return

    cols = 5
    rows = (num_characters // cols) + int(num_characters % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle('Detected Characters', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_characters:
            ax.imshow(characters[i], cmap='gray')
            ax.set_title(f'Character: {labels[i]}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Character grid saved to {save_path}")
    plt.close(fig)

# Function to process each image, segment characters, recognize text, and calculate metrics
def process_image(image_path, grid_save_folder, display_images=False, save_characters=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    img_threshold = img  # Assuming input images are already binarized

    white = [cv2.countNonZero(img_threshold[:, i]) for i in range(width)]
    black = [height - w for w in white]  # Black pixel count per column

    white_max = max(white)
    black_max = max(black)

    def find_end(start):
        end = start + 1
        for m in range(start + 1, width - 1):
            if black[m] > segmentation_spacing * black_max:
                end = m
                break
        return end

    characters = []
    labels = []
    n = 1
    start = 1
    end = 2
    predicted_text = ""

    # Extract ground truth from the file name
    ground_truth = os.path.splitext(os.path.basename(image_path))[0]

    while n < width - 1:
        n += 1
        if white[n] > (1 - segmentation_spacing) * white_max:
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:  # Minimum width for character to avoid noise
                character = img_threshold[:, start:end]
                character_resized = cv2.resize(character, (20, 20)).flatten().reshape(1, -1)
                predicted_label = loaded_model.predict(character_resized)
                predicted_text += predicted_label[0]

                characters.append(character)
                labels.append(predicted_label[0])

                if save_characters:
                    character_filename = os.path.join(chars_output_dir, f"{predicted_label[0]}_{n}.png")
                    cv2.imwrite(character_filename, character)

    # Calculate metrics
    cer = Levenshtein.distance(predicted_text, ground_truth) / len(ground_truth)
    war = 1 - Levenshtein.distance(predicted_text, ground_truth) / max(len(predicted_text.split()), len(ground_truth.split()))
    levenshtein_distance = Levenshtein.distance(predicted_text, ground_truth)

    # Save metrics to CSV
    with open(results_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([image_path, ground_truth, predicted_text, cer, war, levenshtein_distance])

    if display_images:
        grid_filename = os.path.join(grid_save_folder, f"character_grid_{os.path.basename(image_path)}.png")
        display_characters_grid(characters, labels, save_path=grid_filename)

    print(f"Recognized text for {image_path}: {predicted_text}")
    print(f"CER: {cer}, WAR: {war}, Levenshtein Distance: {levenshtein_distance}")

# Initialize the results CSV file with headers
with open(results_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image Path', 'Ground Truth', 'Predicted Text', 'CER', 'WAR', 'Levenshtein Distance'])

# Loop through all images in the input folder and process each one
input_folder = "extracted_plates"
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, grids_output_dir, display_images=True, save_characters=False)

# Calculate the average metrics across all samples
cer_list, war_list, levenshtein_list = [], [], []

with open(results_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header
    for row in csv_reader:
        cer_list.append(float(row[3]))
        war_list.append(float(row[4]))
        levenshtein_list.append(int(row[5]))

average_cer = np.mean(cer_list)
average_war = np.mean(war_list)
average_levenshtein = np.mean(levenshtein_list)

print(f"\nAverage CER: {average_cer}")
print(f"Average WAR: {average_war}")
print(f"Average Levenshtein Distance: {average_levenshtein}")






