import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load the YOLOv8 model for book detection
# Replace 'yolov8n.pt' with the path to your trained YOLO model if custom-trained
yolo_model = YOLO("yolov8n.pt")

# COCO class ID for 'book'
BOOK_CLASS_ID = 73

# Function to detect books using YOLOv8
def detect_book_with_yolo(image):
    """
    Detect books in the given image using YOLOv8.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        bool: True if a book is detected, False otherwise.
        numpy.ndarray: Image with detection annotations.
    """
    results = yolo_model(image)  # Run YOLOv8 detection
    annotated_image = image.copy()

    # Process detections
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == BOOK_CLASS_ID:  # Check if the detected class is 'book'
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                # Draw bounding box and label
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_image,
                    "Book",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                return True, annotated_image  # Return True and annotated image
    return False, annotated_image  # No book detected

# Main function to test book detection
def main():
    # Open the camera (change index if you use an external camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting book detection... Press 'ESC' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect book in the current frame
        book_detected, annotated_frame = detect_book_with_yolo(frame)

        # Display results
        if book_detected:
            print("Book detected!")
        else:
            print("No book detected.")

        # Show the annotated frame
        cv2.imshow("Book Detection", annotated_frame)

        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
