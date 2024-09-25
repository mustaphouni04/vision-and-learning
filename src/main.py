from template_detection import *

templates = ['../template.png', '../template_lateral.png']

result, images = detect_using_template(images_folder_path = '../../images/Test', 
                      template_paths = templates, 
                                       threshold = 0.4) # threshold can be updated to minimize errors

show_detection(result)

calculate_accuracy(images_folder_path = '../../images/Test', 
                      template_paths = templates, 
                   threshold = 0.4)


