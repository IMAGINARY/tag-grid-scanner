# loosely based on https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615
import sys
from copy import deepcopy

import cv2
import numpy as np
import json
import time

from taggridscanner.aux.config import store_config, set_calibration
from taggridscanner.aux.threading import WorkerThread, ThreadSafeContainer
from taggridscanner.aux.utils import Functor
from taggridscanner.pipeline.generate_calibration_pattern import (
    GenerateCalibrationPattern,
)
from taggridscanner.pipeline.retrieve_image import RetrieveImage
from taggridscanner.pipeline.view_image import ViewImage


class CalibrateWorker(Functor):
    def __init__(self, args):
        super().__init__()
        config_with_defaults = args["config-with-defaults"]

        self.retrieve_image = RetrieveImage.create_from_config(config_with_defaults)

        self.checkerboard = (args["rows"], args["cols"])

        self.error_tolerance = args["tolerance"]

        # stop the iteration when specified
        # accuracy, epsilon, is reached or
        # specified number of iterations are completed.
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Vector for 3D points
        self.threedpoints = []

        # Vector for 2D points
        self.twodpoints = []

        #  3D points real world coordinates
        self.objectp3d = np.zeros(
            (1, self.checkerboard[0] * self.checkerboard[1], 3), np.float32
        )
        self.objectp3d[0, :, :2] = np.mgrid[
            0 : self.checkerboard[0], 0 : self.checkerboard[1]
        ].T.reshape(-1, 2)

        self.good_frame_ts = time.perf_counter()
        self.last_error = float("inf")

    def __call__(self):
        frame = self.retrieve_image()

        grayColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            grayColor,
            self.checkerboard,
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
                grayColor, corners, (11, 11), (-1, -1), self.criteria
            )

            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame, self.checkerboard, corners2, ret)

            error, *_ = cv2.calibrateCamera(
                [self.objectp3d], [corners2], grayColor.shape[::-1], None, None
            )

            print("{} (tolerance: {})".format(error, self.error_tolerance))

            ts = time.perf_counter()
            if ts - self.good_frame_ts > 2 and error < self.error_tolerance:
                # keep checkerboard coordinates
                self.good_frame_ts = ts
                self.threedpoints.append(self.objectp3d)
                self.twodpoints.append(corners2)
                print(
                    "{} frames captured for calibration (error: {})".format(
                        len(self.twodpoints), error
                    )
                )
        return (
            frame,
            deepcopy(self.threedpoints),
            deepcopy(self.twodpoints),
        )


def calibrate(args):
    num_frames = args["n"]
    generate_calibration_pattern = GenerateCalibrationPattern(
        img_shape=(args["height"], args["width"]),
        pattern_shape=(args["rows"], args["cols"]),
    )

    if not args["no_pattern"]:
        view_pattern = ViewImage("calibration patter")
        (generate_calibration_pattern | view_pattern)()
        cv2.pollKey()

    view_calibration = ViewImage("image to calibrate")

    calibrate_worker = CalibrateWorker(args)
    producer = WorkerThread(calibrate_worker)
    producer.start()
    producer.result.wait()

    view_calibration(
        np.zeros((1, 1), dtype=np.uint8)
    )  # use dummy image, just to have a window for waitKey()
    while True:
        try:
            (frame, threedpoints, twodpoints) = producer.result.get_nowait()
            view_calibration(frame)
            if len(twodpoints) >= num_frames:
                producer.stop()
                break
        except ThreadSafeContainer.Empty:
            pass
        key = cv2.waitKey(1000 // 120)
        if key == 27:  # ESC
            print("Aborting.", file=sys.stderr)
            sys.exit(1)

    (h, w, *_) = frame.shape
    ret, camera_matrix, [distortion], r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, (w, h), None, None
    )

    res_matrix = np.array([[1 / w, 0, 0], [0, 1 / h, 0], [0, 0, 1]])
    rel_camera_matrix = np.matmul(res_matrix, camera_matrix)

    # Display required output
    print(" Camera matrix:")
    print(json.dumps(rel_camera_matrix.tolist()))

    print("\n Distortion coefficient:")
    print(json.dumps(distortion.tolist()))

    undistorted = cv2.undistort(frame, camera_matrix, distortion, None, None)
    view_calibration.title = "calibration result"
    view_calibration(undistorted)

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
