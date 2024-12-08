## MAE 421 - Final Project
## Group 3 Members:
## Med Coulibaly Sylla
## Youngseo Lee
## Shruti Roy
## Daniela Vita

# This script controls the Crazyflie drone to navigate an obstacle course autonomously
# using a Keras/TensorFlow Lite model for obstacle detection and a YOLOv8 model for book detection.

import tensorflow as tf
from cflib.crazyflie import Crazyflie
import cflib.crtp
from ultralytics import YOLO
import time
import numpy as np
import cv2
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

cflib.crtp.init_drivers()

radio_uri = "radio://0/21/2M"

min_y_pos = -1.0
max_y_pos = 1.0

# Load the Keras-trained model for obstacle detection
obstacle_model = tf.keras.models.load_model("obstacle_model_300x300.keras")

# Load the YOLOv8 model (from Coco Dataset) for bookdetection
yolo_model = YOLO("yolov8n.pt")

# Preprocess image for the Keras model
def preprocess_image(image, image_size=(300, 300)):
    image = cv2.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# Detect book using YOLOv8
def detect_book_with_yolo(image):
    results = yolo_model(image)
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 73:  # COCO class ID for 'book'
                width = abs(box[2] - box[0])
                if width > 100:  # Threshold for book proximity
                    return True  # Book detected
    return False  # No book detected

# Check contours for obstacle detection
def check_contours(frame):
    lb1, ub1 = (145, 35, 75), (180, 255, 255)
    lb2, ub2 = (0, 75, 75), (20, 255, 255)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lb1, ub1)
    mask2 = cv2.inRange(hsv, lb2, ub2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_area, largest_contour_index = find_greatest_contour(contours)
    if largest_contour_index != -1 and largest_area > 100:
        contour_x = np.mean(contours[largest_contour_index], axis=0).flatten()[0]
        return mask, True, contour_x
    return mask, False, -1

# Helper function to find the largest contour
def find_greatest_contour(contours):
    largest_area = 0
    largest_index = -1
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_index = i
    return largest_area, largest_index

# Adjust position gradually
def adjust_position(cf, current_x, current_y, direction):
    steps_per_meter = 10
    for _ in range(1):  # Adjust step size if needed
        if direction == "RIGHT":
            current_y -= 1.0 / steps_per_meter
        elif direction == "FORWARD":
            current_x += 1.0 / steps_per_meter
        elif direction == "LEFT":
            current_y += 1.0 / steps_per_meter
        cf.commander.send_position_setpoint(current_x, current_y, 1.0, 0.0)
        time.sleep(0.1)
    return current_x, current_y

def main():
    try:
        with SyncCrazyflie(radio_uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            cf = scf.cf

            # Initialize and stabilize
            cf.param.set_value('stabilizer.controller', '1')
            cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            cf.param.set_value('kalman.resetEstimation', '0')
            time.sleep(2)

            # Ascend to 1 meter and hover for stabilization
            print("Ascending to 1 meter...")
            for z in np.linspace(0, 1.0, num=20):
                cf.commander.send_hover_setpoint(0, 0, 0, z)
                time.sleep(0.1)
            print("Hovering at 1 meter...")
            for _ in range(30):  # Hover for 3 seconds
                cf.commander.send_hover_setpoint(0, 0, 0, 1.0)
                time.sleep(0.1)

            # Open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return

            current_x, current_y = 0, 0
            print("Starting navigation...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check for book
                if detect_book_with_yolo(frame):
                    print("Book detected! Landing.")
                    for z in np.linspace(1.0, 0, num=20):
                        cf.commander.send_hover_setpoint(0, 0, 0, z)
                        time.sleep(0.1)
                    cf.commander.send_stop_setpoint()
                    break

                # Check contours
                mask, obstacle_detected, contour_x = check_contours(frame)
                if current_y < min_y_pos:
                    print("Boundary too far right. Moving LEFT.")
                    current_x, current_y = adjust_position(cf, current_x, current_y, "LEFT")
                elif current_y > max_y_pos:
                    print("Boundary too far left. Moving RIGHT.")
                    current_x, current_y = adjust_position(cf, current_x, current_y, "RIGHT")
                elif obstacle_detected and abs(contour_x - frame.shape[1] // 2) < 100:
                    print("Obstacle too close! Adjusting.")
                    direction = "RIGHT" if contour_x < frame.shape[1] // 2 else "LEFT"
                    current_x, current_y = adjust_position(cf, current_x, current_y, direction)
                else:
                    print("Path clear. Moving FORWARD.")
                    current_x, current_y = adjust_position(cf, current_x, current_y, "FORWARD")

                cv2.imshow("Obstacle Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        cf.commander.send_stop_setpoint()
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()