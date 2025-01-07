import cv2
import numpy as np

def main():
    # Open a connection to the webcam (use 0 for the default camera)
    i = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        if i%10 == 0:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 1)

            # Detect corners using Shi-Tomasi corner detection
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.3, minDistance=30
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

            # Display the shape on the frame
            cv2.putText(
                frame,
                f"Shape: {detected_shape}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Display the frame with detected corners
        cv2.imshow('Shi-Tomasi Corners Detection', frame)
        i += 1

        # Exit the loop when the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
