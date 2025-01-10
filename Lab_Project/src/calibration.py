from typing import List
import numpy as np
import imageio
import cv2
import copy
import os

def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]

def show_image(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
    
def write_image(i, img, folder):
	cv2.imwrite(f'../output/{folder}/' + str(i) + '.jpg', img) 
	
def get_chessboard_points(chessboard_shape, dx, dy):
	points = []
	for i in range(chessboard_shape[1]):
		for j in range(chessboard_shape[0]):
			points.append([i*dx, j*dy, 0])
	return np.array(points, np.float32)

path = '../data/calibration_images/'
imgs_path = [path + file_name for file_name in os.listdir(path)]
imgs = load_images(imgs_path)
imgs_copy = copy.deepcopy(imgs)

corners = [cv2.findChessboardCorners(img, (7, 7)) for img in imgs]
corners_for_calibration = [corner[1] for corner in corners]

imgs_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]
imgs_copy_gray = copy.deepcopy(imgs)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

imgs_corners = [cv2.drawChessboardCorners(img, (7, 7), cor[1], cor[0]) for img, cor in zip(imgs_copy_gray, corners)]

for i, img in enumerate(imgs_corners):    
	show_image(img)
	write_image(i, img, 'corners_detected')
	
chessboard_points = [get_chessboard_points((7, 7), 30, 30)] * len(imgs_gray)
	
rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, corners_for_calibration, imgs_gray[0].shape[::-1], np.zeros((3,3)), np.zeros((1,4)))

# Obtain extrinsics
extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

# Print outputs
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)
print("Extrinsics:\n", extrinsics)

with open("../output/calibration_results.txt", "w") as file:
    file.write("Intrinsics:\n")
    file.write(f"{intrinsics}\n\n")
    
    file.write("Distortion coefficients:\n")
    file.write(f"{dist_coeffs}\n\n")
    
    file.write("Root mean squared reprojection error:\n")
    file.write(f"{rms}\n\n")
    
    file.write("Extrinsics:\n")
    file.write(f"{extrinsics}\n")