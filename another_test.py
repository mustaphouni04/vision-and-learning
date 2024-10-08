import easyocr
import cv2
import matplotlib.pyplot as plt
import joblib


# Load the image
image_path = 'warped_test/warped_6773FDH_jpg.rf.1e551eb6448f0d2daeea474c31828f3c.jpg'
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not load the image. Please check the file path.")
else:
    # Convert the image to RGB as EasyOCR expects images in this format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])

    # Detect text regions using the RGB image
    results = reader.readtext(image_rgb, detail=1)

    # Create a list to store individual character images
    character_images = []

    for (bbox, text, _) in results:
        # Get bounding box coordinates for the entire detected text
        (top_left, top_right, bottom_right, bottom_left) = bbox
        min_x = int(min(top_left[0], bottom_left[0]))
        max_x = int(max(top_right[0], bottom_right[0]))
        min_y = int(min(top_left[1], top_right[1]))
        max_y = int(max(bottom_left[1], bottom_right[1]))

        # Crop the entire detected text region
        cropped_text_region = image[min_y:max_y, min_x:max_x]

        # Loop through each character in the detected text
        for i, char in enumerate(text):
            # Calculate character width based on the overall width of the text
            char_width = (max_x - min_x) // len(text)

            # Define the start and end positions for the character
            char_x_start = min_x + char_width * i
            char_x_end = char_x_start + char_width

            # Ensure we don't exceed the cropped region's boundaries
            char_x_start = max(min_x, char_x_start)
            char_x_end = min(max_x, char_x_end)

            # Crop the individual character from the text region
            char_cropped = cropped_text_region[:, char_x_start - min_x:char_x_end - min_x]

            # Check if the character crop is not empty before adding
            if char_cropped.size > 0:
                character_images.append((char_cropped, char))
            else:
                print(f"Empty character crop for: {char} at index {i}")

    # Display the cropped characters using Matplotlib
    num_characters = len(character_images)
    fig, axes = plt.subplots(1, num_characters, figsize=(15, 5))

    # If there's only one character detected, ensure axes is a list
    if num_characters == 1:
        axes = [axes]

    for idx, (char_image, char) in enumerate(character_images):
        # Check if char_image is empty before attempting to convert
        if char_image.size > 0:
            axes[idx].imshow(cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"Character: {char}")
            axes[idx].axis('off')
            character_resized = cv2.resize(char_image, (32,32))
        else:
            print(f"Skipping empty image for character: {char}")

    plt.tight_layout()
    plt.show()
