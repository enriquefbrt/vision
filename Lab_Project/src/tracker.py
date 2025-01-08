import cv2
import numpy as np
from screeninfo import get_monitors

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify a video file path

# Create Background Subtractor using Gaussian Mixture Model (GMM)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Set up the Kalman filter
kalman = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements (x, y)
kalman.transitionMatrix = np.eye(4, dtype=np.float32)  # State transition matrix
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)  # Measurement matrix
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2  # Process noise covariance
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1  # Measurement noise covariance
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1  # Initial estimation error covariance

# Initialize the state vector (x, y, delta_x, delta_y)
kalman.statePost = np.zeros((4, 1), dtype=np.float32)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Use morphological operations to clean the mask (remove small noise)
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the detected moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours are found
    if contours:
        # Find the largest contour (most likely the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Kalman correction step: Update the measurement with the detected center position
        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        kalman.correct(measurement)

        # Kalman prediction step: Predict the next position
        predicted = kalman.predict()

        # Get the predicted coordinates (next predicted position)
        predicted_x, predicted_y = int(predicted[0]), int(predicted[1])

        # Draw the predicted position and the actual detected position
        cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 255, 255), -1)  # Predicted position
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Actual bounding box
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Actual center

        # Display the coordinates
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"({predicted_x}, {predicted_y})", (predicted_x + 10, predicted_y - 10),
                    font, 0.6, (0, 255, 255), 2)
        
        

    # Show the frame with the tracking information
    cv2.imshow('Object Tracker with Kalman Filter and Background Subtraction', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


