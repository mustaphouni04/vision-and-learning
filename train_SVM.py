import os
import cv2
import numpy as np
import joblib  # For saving and loading the model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


data_path = 'character_data' 

# Parameters
image_size = (32, 32)  # Resize images to a fixed size

# Prepare the data
X = []  # Feature vectors
y = []  # Labels

# Load images and labels
for label in os.listdir(data_path):
    label_path = os.path.join(data_path, label)
    if os.path.isdir(label_path):
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
            if image is not None:
                image = cv2.resize(image, image_size).flatten()  # Resize and flatten the image
                X.append(image)
                y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'svm_character_recognition_model.pkl'
joblib.dump(svm_classifier, model_filename)
print(f"Model saved as {model_filename}")

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Get detailed classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Load the saved model for inference
loaded_model = joblib.load(model_filename)

# Inference example (replace 'sample_image_path' with the path to your test image)
sample_image_path = 'character_data/0/0_3.png'
sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
sample_image_resized = cv2.resize(sample_image, image_size).flatten().reshape(1, -1)
predicted_label = loaded_model.predict(sample_image_resized)
print(f"Predicted label for the sample image: {predicted_label[0]}")