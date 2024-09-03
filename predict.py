import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle  # Use pickle if the model was saved with pickle


# If using pickle to load the model:
with open('bone_age_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
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

# Load the validation dataset
val_df = pd.read_csv('./Bone Age Validation Set/Validation Dataset.csv')  # Modify the CSV file path accordingly

# Get image IDs and bone ages from the validation set
val_image_ids = val_df['id'].values
val_bone_ages = val_df['boneage'].values

# Load and preprocess validation images
X_val = load_and_preprocess_images(val_image_ids, './Bone Age Validation Set/boneage-validation')
X_val = X_val.reshape(X_val.shape[0], 224, 224, 1)  # Reshape for grayscale images
y_val = val_bone_ages[:X_val.shape[0]]

# Predict bone ages for the validation set
y_pred = model.predict(X_val)

# Calculate metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)

print(f"Validation Mean Absolute Error (MAE): {mae}")
print(f"Validation Mean Squared Error (MSE): {mse}")

# Plotting graphs
plt.figure(figsize=(12, 6))

# Plot 1: Actual vs Predicted Bone Age
plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
plt.xlabel('Actual Bone Age')
plt.ylabel('Predicted Bone Age')
plt.title('Actual vs Predicted Bone Age')

# Plot 2: Error Distribution
plt.subplot(1, 2, 2)
errors = y_val - y_pred.flatten()
plt.hist(errors, bins=30, color='blue', alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Error Distribution')

plt.tight_layout()
plt.show()

# Optional: Display a few sample predictions
n_samples = 5
sample_indices = np.random.choice(range(len(y_val)), n_samples, replace=False)
plt.figure(figsize=(15, 5))
for i, idx in enumerate(sample_indices):
    plt.subplot(1, n_samples, i + 1)
    plt.imshow(X_val[idx].reshape(224, 224), cmap='gray')
    plt.title(f"Actual: {y_val[idx]}\nPredicted: {y_pred[idx][0]:.2f}")
    plt.axis('off')

plt.show()
