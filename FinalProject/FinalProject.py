## MAE 421 - Final Project
## Group 3 Members:
## Med Coulibaly Sylla
## Youngseo Lee
## Shruti Roy
## Daniela Vita

# This script controls the Crazyflie drone to navigate an obstacle course autonomously
# using a Keras/TensorFlow Lite model for obstacle detection and a YOLOv8 model for book detection.

# Need to do:
# 1. Implement the area boundary code
# 2. Improve the book detection
# 3. Tune the navigation + object avoidance

import cv2
import numpy as np
import tensorflow as tf
from cflib.crazyflie import Crazyflie
import cflib.crtp
import time
from ultralytics import YOLO

# Initialize the Crazyflie communication drivers
cflib.crtp.init_drivers()

# Define the radio URI for your Crazyflie
radio_uri = "radio://0/3/2M"

# Load the Keras-trained model for obstacle detection
# Updated to use the 300x300 model
obstacle_model = tf.keras.models.load_model("obstacle_model_300x300.keras")

# Load the YOLOv8 model (from Coco Dataset) for book detection
yolo_model = YOLO("yolov8n.pt")

# Initialize Crazyflie
cf = Crazyflie(rw_cache="./cache")

# Preprocess image for Keras model
def preprocess_image(image, image_size=(300, 300)):
    """
    Preprocess the input image for the obstacle detection model.

    Args:
        image (numpy.ndarray): Input image.
        image_size (tuple): Target size for resizing.

    Returns:
        numpy.ndarray: Preprocessed image ready for the model.
    """
    image = cv2.resize(image, image_size)  # Resize to match the model input size
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)

# Detect book using YOLOv8
def detect_book_with_yolo(image):
    """
    Detect books in the given image using YOLOv8.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        bool: True if a book is detected, False otherwise.
    """
    results = yolo_model(image)  # Run YOLOv8 detection
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 73:  # COCO class ID for 'book'
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image, "Book", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )
                return True  # Book detected
    return False  # No book detected

# Command Crazyflie to hover, move, or land
def send_command(cf, command):
    """
    Send movement commands to the Crazyflie.

    Args:
        cf (Crazyflie): Crazyflie object.
        command (str): Command to execute.
    """
    if command == "STOP":
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.0)  # Stop movement
    elif command == "RIGHT":
        cf.commander.send_hover_setpoint(0.0, 0.3, 0.5, 0.5)  # Move right
    elif command == "FORWARD":
        cf.commander.send_hover_setpoint(0.5, 0.0, 0.5, 0.5)  # Move forward
    elif command == "LAND":
        cf.commander.send_stop_setpoint()  # Land

# Process frames and send commands
def process_frames(cf, cap):
    """
    Process video frames for obstacle and book detection.

    Args:
        cf (Crazyflie): Crazyflie object.
        cap (cv2.VideoCapture): Video capture object.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Check for the book
        if detect_book_with_yolo(frame):
            print("Book detected! Landing.")
            send_command(cf, "LAND")
            break

        # Preprocess the frame for obstacle prediction
        input_data = preprocess_image(frame)
        prediction = obstacle_model.predict(input_data)[0][0]

        # Process prediction and send command
        if prediction > 0.5:  # Obstacle detected
            print("Obstacle detected! Moving right.")
            send_command(cf, "RIGHT")
        else:  # No obstacle
            print("No obstacle. Moving forward.")
            send_command(cf, "FORWARD")

        # Display the frame with predictions
        cv2.putText(frame, f"Obstacle: {'Yes' if prediction > 0.5 else 'No'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Book: {'Yes' if detect_book_with_yolo(frame) else 'No'}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Obstacle and Book Detection", frame)

        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Main control function
def main():
    try:
        # Connect to Crazyflie
        cf.open_link(radio_uri)
        print("Connected to Crazyflie!")

        # Wait for the Crazyflie to initialize
        time.sleep(2)

        # Take off
        print("Taking off...")
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.5, 0.0)  # Ascend to 0.5 meters
        time.sleep(2)

        # Open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Starting obstacle and book detection...")
        process_frames(cf, cap)

        # Land the Crazyflie
        print("Landing...")
        send_command(cf, "LAND")
        time.sleep(2)

    except KeyboardInterrupt:
        print("Interrupted! Landing...")
        send_command(cf, "LAND")
    finally:
        cf.close_link()
        print("Connection closed.")
        cv2.destroyAllWindows()

# Run the script
if __name__ == "__main__":
    main()
