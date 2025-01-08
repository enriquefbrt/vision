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
    
    if not data:
        return None  


    frequency = {}
    for item in data:
        if item in frequency:
            frequency[item] += 1
        else:
            frequency[item] = 1


    max_count = 0
    for count in frequency.values():
        if count > max_count:
            max_count = count

    modes = []
    for key, value in frequency.items():
        if value == max_count:
            modes.append(key)


    return modes[0]

def security_system():
    global progress
   
    i = 0
    
    interval_corners = []
    guess = []

    while True:
      
        frame = picam.capture_array()


        if i%10 == 0:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            
            gray = cv2.GaussianBlur(gray, (5, 5), 1)

            
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.15, minDistance=30
            )

        if corners is not None:
            
            corners = np.int0(corners)

            
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            
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

            
            cv2.putText(
                frame,
                f"Image {len(guess) + 1}, Percentage: {int((i%frame_count)/frame_count*100)}%, Shape: {detected_shape}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                1
            )

        
        cv2.imshow('Shi-Tomasi Corners Detection', frame)
        i += 1

        # Exit the loop when the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

security_system()

pygame.init()

BACKGROUND_COLOR = (0, 0, 0)
BRUSH_COLOR = (255, 255, 255)

window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Click to Paint")

running = True
mouse_pressed = False

bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

kalman = cv2.KalmanFilter(4, 2)  
kalman.transitionMatrix = np.eye(4, dtype=np.float32) 
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)  
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2  
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1  
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1  


kalman.statePost = np.zeros((4, 1), dtype=np.float32)

screen_width = 1920
screen_heigh = 1080
prev_point = None

while True:
    frame = picam.capture_array()
    frame = cv2.flip(frame, -1)


    fg_mask = bg_subtractor.apply(frame)


    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        largest_contour = max(contours, key=cv2.contourArea)


        x, y, w, h = cv2.boundingRect(largest_contour)


        center_x = x + w // 2
        center_y = y + h // 2


        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        kalman.correct(measurement)


        predicted = kalman.predict()

        
        predicted_x, predicted_y = int(predicted[0]), int(predicted[1])

        cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 255, 255), -1) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  

     
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
                if event.key == pygame.K_c:  
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

    cv2.imshow('Object Tracker with Kalman Filter and Background Subtraction', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
