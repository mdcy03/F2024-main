import cv2


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Start with index 0

if not cap.isOpened():
    print("Camera index 0 not available, trying index 1.")
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # Try the next index

if not cap.isOpened():
    print("No available camera found.")
else:
    print("Camera opened successfully!")
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
