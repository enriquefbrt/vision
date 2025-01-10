# 📷 Camera Calibration, Pattern Detection, and Object Tracking on Raspberry Pi

This project implements a camera calibration tool, pattern detection for a visual password system, and object tracking using OpenCV and Python on a Raspberry Pi with a PiCamera. The object tracker is integrated into a fun "paint" application, allowing you to draw with your hand or any object!

## 🌟 Features

### 1. 🛠️ Camera Calibration

- Use chessboard images to calibrate the camera and remove distortion.
- Computes camera matrix and distortion coefficients.

### 2. 🎨 Pattern Detection (Visual Password System)

- Detects geometric shapes such as squares, rectangles, and circles.
- Enables a "visual password" system where the user shows a sequence of shapes as the password.
- Put each shape into the camera until the percentage reaches 100%, if the password is correct, you will move on.

### 3. 🔠 Object Tracking with Paint Application

- Tracks moving objects using:
  - **Kalman Filter**: Predicts the motion of the object.
  - **MOG (Mixture of Gaussians) Background Subtraction**: Isolates moving objects from the background.
- Integrated into a "paint" application where you can draw on the screen by moving an object in front of the camera. Click left Mousebutton to draw.

## 📊 Requirements

To run this project, you will need the following:

- A Raspberry Pi (tested on Raspberry Pi 3/4).
- A PiCamera.
- Python 3 installed on the Raspberry Pi.
- OpenCV library.
- Picamera module.

## ✨ Notes

- Ensure good lighting conditions for accurate pattern detection and object tracking.
- Calibrate the camera before running other scripts for better accuracy.


## 📚 Contributions

Contributions, issues, and feature requests are welcome! Feel free to fork this repository and submit a pull request.

## 🙏 Acknowledgments

- OpenCV documentation and tutorials for computer vision techniques.
- The Raspberry Pi Foundation for providing affordable and versatile computing platforms.



Happy painting! 🚀
