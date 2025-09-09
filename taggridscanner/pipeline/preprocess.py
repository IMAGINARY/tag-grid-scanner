import cv2
import numpy as np

from taggridscanner.aux.utils import Functor


class Preprocess(Functor):
    def __init__(self, rel_camera_matrix, distortion_coefficients, rotate, flip_h, flip_v):
        super().__init__()

        self.correct_distortion = create_distortion_corrector(rel_camera_matrix, distortion_coefficients)
        self.linear_transform = create_linear_transformer(rotate, flip_h, flip_v)

    def __call__(self, image):
        return self.linear_transform(self.correct_distortion(image))

    @staticmethod
    def create_from_config(config):
        camera_config = config["camera"]

        calibration_config = camera_config["calibration"]
        rel_camera_matrix, distortion_coefficients = (
            np.array(calibration_config["matrix"]),
            np.array(calibration_config["distortion"]),
        )
        rotate, flip_h, flip_v = (
            camera_config["rotate"],
            camera_config["flipH"],
            camera_config["flipV"],
        )
        return Preprocess(rel_camera_matrix, distortion_coefficients, rotate, flip_h, flip_v)


def get_rotate_code(degrees):
    rotate_codes = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return rotate_codes.get(degrees)


def get_flip_code(flip_h, flip_v):
    if flip_v and not flip_h:
        return 0
    elif not flip_v and flip_h:
        return 1
    elif flip_v and flip_h:
        return -1
    else:
        return None


def create_linear_transformer(rotate, flip_h, flip_v):
    rotate_code = get_rotate_code(rotate)
    flip_code = get_flip_code(flip_h, flip_v)

    def linear_transformer(img):
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)
        if flip_code is not None:
            img = cv2.flip(img, flip_code)
        return img

    return linear_transformer


def create_inverse_linear_transformer(rotate, flip_h, flip_v):
    rotate_code = get_rotate_code((360 - rotate) % 360)
    flip_code = get_flip_code(flip_h, flip_v)

    def inverse_linear_transformer(img):
        if flip_code is not None:
            img = cv2.flip(img, flip_code)
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)
        return img

    return inverse_linear_transformer


def create_distortion_corrector(rel_camera_matrix, distortion_coefficients):
    if rel_camera_matrix is not None and distortion_coefficients is not None:
        h, w, map_x, map_y = None, None, None, None

        def undistort(img):
            nonlocal h, w, map_x, map_y
            if (h, w) != img.shape[0:2] or map_x is None or map_y is None:
                h, w = img.shape[0:2]
                res_matrix = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]])
                abs_camera_matrix = np.matmul(res_matrix, rel_camera_matrix)
                map_x, map_y = cv2.initUndistortRectifyMap(
                    abs_camera_matrix, distortion_coefficients, None, abs_camera_matrix, (w, h), cv2.CV_32FC1
                )

            return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        return undistort
    else:
        return lambda img: img
