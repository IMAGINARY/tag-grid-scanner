# loosely based on https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615

import cv2
import numpy as np
import time

from utils import save_coefficients

num_frames = 10
last_frame = None 
capture = cv2.VideoCapture(0)

# Define the dimensions of checkerboard
CHECKERBOARD = (6, 9)

# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []


#  3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] 
                      * CHECKERBOARD[1], 
                      3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)

t0 = time.perf_counter()

while(True):
    ret, frame = capture.read()
         
    grayColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(
                grayColor, CHECKERBOARD, 
                cv2.CALIB_CB_ADAPTIVE_THRESH 
                + cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE)
  
    # If desired number of corners can be detected then,
    # refine the pixel coordinates and display
    # them on the images of checker board
    if ret == True:
        # Refining pixel coordinates
        # for given 2d points.
        corners2 = cv2.cornerSubPix(
            grayColor, corners, (11, 11), (-1, -1), criteria)
        
        # Draw and display the corners
        image = cv2.drawChessboardCorners(frame.copy(), 
                                          CHECKERBOARD, 
                                          corners2, ret)
        cv2.imshow('video', image)
        
        t1 = time.perf_counter()
        if t1 - t0 > 2:
            # keep checkerboard coordinates
            last_frame = frame
            t0 = t1
            threedpoints.append(objectp3d)
            twodpoints.append(corners2)
            print("{}/{} frames captured for calibration".format(len(twodpoints), num_frames))
            if len(twodpoints) == num_frames:
                break
        cv2.waitKey(1)
    else:
        cv2.imshow('video', frame)
        cv2.waitKey(1)

ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    # Displayig required output
print(" Camera matrix:")
print(matrix)
  
print("\n Distortion coefficient:")
print(distortion)

undistorted = cv2.undistort(last_frame, matrix, distortion, None, None)
cv2.imshow('video', undistorted)
cv2.waitKey()

capture.release()
cv2.destroyAllWindows()

cv2.imwrite('before.jpg', last_frame)
cv2.imwrite('after.jpg', undistorted)

save_coefficients(matrix, distortion, "camera-profile.yml")
