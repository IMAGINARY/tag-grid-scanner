# loosely based on https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615

import cv2
import numpy as np
import time

from .utils import setup_video_capture, save_calibration_coefficients

num_frames = 1
# Define the dimensions of checkerboard
CHECKERBOARD = (17, 31)


def compute_error(obj_points, img_points, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error
    return mean_error / len(obj_points)


def calibrate(args, config, config_with_defaults):
    camera_config = config_with_defaults["camera"]
    profile_path = camera_config["calibration"]

    last_frame = None
    capture = setup_video_capture(camera_config)

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
        -1, 2
    )

    t0 = time.perf_counter()
    last_error = float("inf")

    while True:
        ret, frame = capture.read()

        if not ret:
            continue

        grayColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            grayColor,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria
            )

            # Draw and display the corners
            image = cv2.drawChessboardCorners(frame.copy(), CHECKERBOARD, corners2, ret)
            cv2.imshow("video", image)

            tmp_threed_points = threedpoints + [objectp3d]
            tmp_twod_points = twodpoints + [corners2]

            ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
                tmp_threed_points, tmp_twod_points, grayColor.shape[::-1], None, None
            )

            error = compute_error(
                tmp_threed_points, tmp_twod_points, r_vecs, t_vecs, matrix, distortion
            )
            print(error)

            t1 = time.perf_counter()
            if t1 - t0 > 2 and error < 0.0125:
                print(error, last_error)
                # keep checkerboard coordinates
                last_frame = frame
                last_error = error
                t0 = t1
                threedpoints.append(objectp3d)
                twodpoints.append(corners2)

                print(
                    "{}/{} frames captured for calibration (error: {})".format(
                        len(twodpoints), num_frames, error
                    )
                )
                if len(twodpoints) == num_frames:
                    break
            cv2.waitKey(1)
        else:
            cv2.imshow("video", frame)
            cv2.waitKey(1)

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None
    )

    # Displayig required output
    print(" Camera matrix:")
    print(matrix)

    print("\n Distortion coefficient:")
    print(distortion)

    undistorted = cv2.undistort(last_frame, matrix, distortion, None, None)
    cv2.imshow("video", undistorted)
    cv2.waitKey()

    capture.release()
    cv2.destroyAllWindows()

    print("Saving profile to {}".format(profile_path))
    save_calibration_coefficients(matrix, distortion, profile_path)
