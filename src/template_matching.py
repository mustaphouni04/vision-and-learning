from utils_template import *

templates = ['../img/template.png']

result, images = detect_using_template(images_folder_path = '../original_images', 
                      template_paths = templates, 
                                       threshold = 0.1) # threshold can be updated to minimize errors

show_detection(result)

calculate_accuracy(images_folder_path = '../original_images', 
                      template_paths = templates, 
                   threshold = 0.1)

