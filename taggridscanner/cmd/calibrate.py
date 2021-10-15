# loosely based on https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615

import cv2
import numpy as np
import json
import time

from taggridscanner.aux.config import store_config, set_calibration
from taggridscanner.pipeline.retrieve_image import RetrieveImage

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


def calibrate(args):
    config_with_defaults = args["config-with-defaults"]

    last_frame = None
    retrieve_image = RetrieveImage.create_from_config(config_with_defaults)

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
        frame = retrieve_image()

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
        if ret:
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

    ret, camera_matrix, [distortion], r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None
    )

    (h, w) = retrieve_image.size
    res_matrix = np.array([[1 / w, 0, 0], [0, 1 / h, 0], [0, 0, 1]])
    rel_camera_matrix = np.matmul(res_matrix, camera_matrix)

    # Display required output
    print(" Camera matrix:")
    print(json.dumps(rel_camera_matrix.tolist()))

    print("\n Distortion coefficient:")
    print(json.dumps(distortion.tolist()))

    undistorted = cv2.undistort(last_frame, camera_matrix, distortion, None, None)
    cv2.imshow("video", undistorted)

    config_path = args["config-path"]
    print(
        "Press ENTER to save calibration profile to config file: {}".format(config_path)
    )
    print("Press any other key to abort.")
    key = cv2.waitKey()

    cv2.destroyAllWindows()

    if key == 13:  # <ENTER>
        print("Saving calibration profile to: {}".format(config_path))
        modified_raw_config = set_calibration(
            args["raw-config"], rel_camera_matrix, distortion
        )
        store_config(modified_raw_config, config_path)
    else:
        print("Aborting.")
