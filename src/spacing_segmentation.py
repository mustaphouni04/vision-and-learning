import cv2
import pytesseract
import joblib

# Ordinary license plate value is 0.95, new energy license plate is changed to 0.9
segmentation_spacing = 0.92

'''1 read the picture, and do grayscale processing'''
img = cv2.imread("../warped_test/warped_3938GSP_jpg.rf.eb8908fed7ad7778a3d3231e4b6a7550.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = cv2.resize(img_gray, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

'''2 Binary the grayscale image'''
ret, img_threshold = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)


'''3 Split characters'''
white = []  # Record the sum of white pixels in each column
black = []  # Record the sum of black pixels in each column
height = img_threshold.shape[0]
width = img_threshold.shape[1]

white_max = 0
black_max = 0


'''4 Cycle through the sum of black and white pixels for each column'''
for i in range(width):
    white_count = 0
    black_count = 0
    for j in range(height):
        if img_threshold[j][i] == 255:
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
loaded_model = joblib.load("../models/svm_character_recognition_model.pkl")
text = ""
while n < width - 1:
    n += 1
    if(white[n] > (1 - segmentation_spacing) * white_max):
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            print(start, end)
            character = img_threshold[1:height, start:end]
            character_resized = cv2.resize(character, (32,32)).flatten().reshape(1, -1)
            predicted_label = loaded_model.predict(character_resized)
            text += predicted_label[0]
            print(f"Predicted label for the sample image: {predicted_label[0]}")
            cv2.imwrite('img/{0}.png'.format(n), character)      
            cv2.imshow('character', character)
            cv2.imshow('image', img_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
print(text)