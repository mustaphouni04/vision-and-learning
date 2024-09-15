import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../images/Test/Frontal/Frontal/1062FNT.jpg', cv2.IMREAD_GRAYSCALE)

image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off')  
plt.show()
