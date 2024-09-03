import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pickle  # Import pickle to save the model

# Load the dataset CSV file
df = pd.read_csv('boneage_with_real_age.csv')

# Function to load and preprocess images
def load_and_preprocess_images(image_ids, img_dir, img_size=(224, 224)):
    images = []
    for img_id in image_ids:
        # Construct the full image path
        img_path = os.path.join(img_dir, f"{img_id}.png")  # Replace with the correct extension if needed
        # Print the ID of the image being processed
        print(f"Loading image with ID: {img_id}")
        # Read the image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Image not found at path: {img_path}. Skipping...")
            continue
        # Resize the image to the required input size for the model
        img = cv2.resize(img, img_size)
        # Normalize the image
        img = img / 255.0
        images.append(img)
    return np.array(images)

# Get the image IDs from the CSV file
image_ids = df['id'].values

# Load and preprocess images
X = load_and_preprocess_images(image_ids, 'boneage-training-dataset')

# Ensure some images were loaded
if len(X) == 0:
    raise ValueError("No images were loaded. Please check the image paths.")

# Reshape for grayscale images
X = X.reshape(X.shape[0], 224, 224, 1)
y = df['boneage'].values[:X.shape[0]]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Image data generator for augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1)  # Output layer for bone age regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    validation_data=val_datagen.flow(X_val, y_val),
    epochs=50
)

# Save the trained model using pickle
with open('bone_age_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)


