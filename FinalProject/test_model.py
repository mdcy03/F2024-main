import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, image_size=(300, 300)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image from path {image_path}")
    # Resize and normalize the image
    image_resized = cv2.resize(image, image_size)
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0).astype(np.float32)
    return image_input, image

# Function to test the Keras model
def test_keras_model(model_path, image_input):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    prediction = model.predict(image_input)[0][0]  # Get prediction
    return prediction

# Function to visualize images
def visualize_images(original_image, preprocessed_image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(preprocessed_image[0])  # Display preprocessed image
    plt.title("Preprocessed Image")
    plt.axis("off")
    plt.show()

# Path to the image and model
image_path = '/Users/med/Library/CloudStorage/OneDrive-PrincetonUniversity/Fall 2024 - Princeton University/MAE 345 - Introduction to Robotics/F2024-main/FinalProject/IMG_7575.jpeg'
model_path = "obstacle_model_300x300.keras"

# Check if paths exist
if not os.path.exists(image_path):
    print(f"Image path does not exist: {image_path}")
elif not os.path.exists(model_path):
    print(f"Model path does not exist: {model_path}")
else:
    try:
        # Load and preprocess the image
        image_input, original_image = load_and_preprocess_image(image_path)

        # Visualize images
        visualize_images(original_image, image_input)

        # Test the model
        prediction = test_keras_model(model_path, image_input)

        # Interpret prediction
        threshold = 0.5
        obstacle_status = "Obstacle" if prediction > threshold else "No Obstacle"
        print(f"Raw Prediction Score: {prediction:.4f} ({obstacle_status})")

    except Exception as e:
        print(f"An error occurred: {e}")

