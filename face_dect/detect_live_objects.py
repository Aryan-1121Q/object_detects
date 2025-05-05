import cv2
import torch

# Load the YOLOv5 model (ensure you have a pretrained model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the Mac's webcam (the first camera is usually 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()

    # If we got the frame, process it
    if ret:
        # Perform object detection on the frame
        results = model(frame)

        # Render the results on the frame (boxes and labels)
        results.render()

        # Display the frame with detected objects
        cv2.imshow('Real-time Object Detection', frame)

    # Break if the user presses 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
