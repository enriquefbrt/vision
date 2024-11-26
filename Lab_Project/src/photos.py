from picamera2 import Picamera2
from time import sleep
import cv2

def capture():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280,720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    
    sleep(2)

    cont = 0
    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cont += 1
            image = picam.capture_array()
            cv2.imshow("Tablero capturado", image)
            cv2.imwrite(f"../data/calibration_images/chessboard{cont}.jpg", image)


if __name__ == "__main__":
    capture()