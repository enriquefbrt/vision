import cv2
import numpy as np

PASSWORD = [4,3,4,5]
progress = 0
frame_count = 400

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



def main():
    global progress
    # Open a connection to the webcam (use 0 for the default camera)
    i = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    interval_corners = []
    guess = []

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
                        print("Incorrect password gilipoyas")

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

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
