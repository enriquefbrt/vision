import cv2
import numpy as np
import pygame
from picamera2 import Picamera2

PASSWORD = [4,3,4,5]
progress = 0
frame_count = 400
picam = Picamera2()
picam.preview_configuration.main.size=(1280, 720)
picam.preview_configuration.main.format="RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()

def find_mode(data):
    """
    Returns the mode(s) of a list. If there are multiple modes, it returns all of them.
    """
    if not data:
        return None  # Return None if the list is empty

    # Count occurrences manually
    frequency = {}
    for item in data:
        if item in frequency:
            frequency[item] += 1
        else:
            frequency[item] = 1

    # Find the maximum frequency
    max_count = 0
    for count in frequency.values():
        if count > max_count:
            max_count = count

    # Find all elements with the maximum frequency
    modes = []
    for key, value in frequency.items():
        if value == max_count:
            modes.append(key)

    # If there's only one mode, return it directly
    return modes[0]

def security_system():
    global progress
    # Open a connection to the webcam (use 0 for the default camera)
    i = 0
    
    interval_corners = []
    guess = []

    while True:
        # Capture frame-by-frame
        frame = picam.capture_array()


        if i%10 == 0:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 1)

            # Detect corners using Shi-Tomasi corner detection
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.15, minDistance=30
            )

        if corners is not None:
            # Convert corners to integer values
            corners = np.int0(corners)

            # Draw circles at each detected corner
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Detect the shape based on the number of corners
            corner_count = len(corners)
            interval_corners.append(corner_count)
            if corner_count == 3:
                detected_shape = "Triangle"
            elif corner_count == 4:
                detected_shape = "Quadrilateral"
            elif corner_count == 5:
                detected_shape = "Pentagon"
            elif corner_count == 6:
                detected_shape = "Hexagon"
            else:
                detected_shape = "Unknown"

            if i%frame_count == 0 and i != 0:
                mode =  find_mode(interval_corners)
                guess.append(mode)
                interval_corners = []
                print(mode)
                
                if len(guess) == len(PASSWORD):
                    if guess == PASSWORD:
                        break
                    else:
                        guess = []
                        print("Incorrect password")

            # Display the shape on the frame
            cv2.putText(
                frame,
                f"Shape: {detected_shape}, {len(guess)} Image, Percentage: {int((i%frame_count)/frame_count*100)}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        # Display the frame with detected corners
        cv2.imshow('Shi-Tomasi Corners Detection', frame)
        i += 1

        # Exit the loop when the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

#security_system()

pygame.init()

BACKGROUND_COLOR = (0, 0, 0)
BRUSH_COLOR = (255, 255, 255)

window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Click to Paint")

running = True
mouse_pressed = False

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

screen_width = 1920
screen_heigh = 1080
prev_point = None

while True:
    frame = picam.capture_array()
    frame = cv2.flip(frame, -1)

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
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_pressed = False
                    prev_point = None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Reset the screen on 'C' key press
                    window.fill(BACKGROUND_COLOR)
                    prev_point = None

        if mouse_pressed:
            if prev_point != None:
                pos = (predicted_x * screen_width // frame.shape[1], predicted_y * screen_heigh // frame.shape[0])
                pygame.draw.line(window, BRUSH_COLOR, prev_point, pos, 5)
                prev_point = pos
            else:
                prev_point = (predicted_x * screen_width // frame.shape[1], predicted_y * screen_heigh // frame.shape[0])

    pygame.display.flip()

    # Show the frame with the tracking information
    cv2.imshow('Object Tracker with Kalman Filter and Background Subtraction', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cv2.destroyAllWindows()
