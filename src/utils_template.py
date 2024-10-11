import cv2
import os
import random
import matplotlib.pyplot as plt

def load_images(images_folder_path, template_path):
    """
    Load images from a folder and return a list of tuples containing the image file name and the image itself.

    Args:
        images_folder_path (str): The path to the folder containing the images.
        template_path (str): This parameter is not used in this function.

    Returns:
        list: A list of tuples containing the image file name and the image itself.
    """
    folder_path = images_folder_path
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
    images = []

    # Calculate the number of images to load (50% of the total)
    num_images_to_load = len(image_files) // 3

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Load the first 'num_images_to_load' images
    for image_file in image_files[:num_images_to_load]:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, 0)
        images.append((image_file, image))

    return images

def detect_using_template(images_folder_path, template_paths, threshold=0.1, width=1920, height=1080):
    """
    Detects the region of the rectangle in the image using template matching.

    Args:
    images_folder_path (str): Path to the folder containing the images.
    template_paths (list): List of paths to the template images.
    threshold (float, optional): Threshold for the maximum value in the result matrix. Defaults to 0.1.
    width (int, optional): Width of the resized image. Defaults to 1920.
    height (int, optional): Height of the resized image. Defaults to 1080.

    Returns:
    list: A list of tuples containing the image file name, the image itself, and the region of the rectangle.
    """

    # load the images
    images = load_images(images_folder_path, template_paths[0])  # Load images using the first template path

    results = []

    # process each image
    for i, (image_file, img) in enumerate(images):
        # Resize the image
        width = width
        height = height
        coord = (width, height)
        img = cv2.resize(img, coord, interpolation=cv2.INTER_AREA)

        # apply thresholding to enhance contrast
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        max_val = 0
        max_template_path = None
        max_template = None
        max_loc = None

        # perform template matching for each template
        for template_path in template_paths:
            # load the template image
            template = cv2.imread(template_path, 0)

            # apply thresholding to enhance contrast
            template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # perform template matching
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

            # find the maximum value in the result matrix
            min_val, val, min_loc, loc = cv2.minMaxLoc(result)

            # check if the maximum value is above the current maximum value
            if val > max_val:
                max_val = val
                max_template_path = template_path
                max_template = template
                max_loc = loc

        # check if the maximum value is above the threshold
        if max_val >= threshold:
            # draw a rectangle around the matched region
            cv2.rectangle(img, max_loc, (max_loc[0] + max_template.shape[1], max_loc[1] + max_template.shape[0]), (0, 255, 0), 2)

            # get the region of the rectangle
            rectangle_region = img[max_loc[1]:max_loc[1] + max_template.shape[0], max_loc[0]:max_loc[0] + max_template.shape[1]]

            # append the result to the list
            results.append((image_file, img, rectangle_region, max_template_path))
        else:
            print(f"No match found for {image_file}")

    return results, images


def show_detection(results):
    for result in results:
        output_path = f"../template_detected/output_{result[0]}"
        cv2.imwrite(output_path, result[2])
        print(f"Image saved to {output_path}")

def calculate_accuracy(images_folder_path, template_paths, threshold=0.1, width=1600, height=900):
    results, images = detect_using_template(images_folder_path, template_paths, threshold=0.1, width=1600, height=900)
    print(len(results), " plates have been detected.")
    print(len(images), " images we have in total.")
    print("Accuracy: ", len(results)/len(images))


    
    