import math
import cv2
import numpy as np
import logging

from taggridscanner.aux.utils import Functor


logger = logging.getLogger(__name__)


class Preprocess(Functor):
    def __init__(self, rel_camera_matrix, distortion_coefficients, scale: tuple[float, float], rotate, flip_h, flip_v):
        assert math.isfinite(scale[0]) and math.isfinite(scale[1])
        assert scale[0] > 0 and scale[1] > 0
        super().__init__()

        if scale[0] > 1.0 > scale[1] or scale[1] > 1.0 > scale[0]:
            m = "Non-uniform scaling with one axis > 1.0 and the other < 1.0 may cause interpolation artifacts."
            logger.warning(m)

        self.__rel_camera_matrix = rel_camera_matrix
        self.__distortion_coefficients = distortion_coefficients
        self.__scale = scale
        self.__rotate = rotate
        self.__flip_h = flip_h
        self.__flip_v = flip_v

        self.operator = None
        self.last_image_shape = None

    def result_size_before_rotation(self, size: tuple[int, int]):
        scaled_height = int(math.ceil(size[0] * self.__scale[0]))
        scaled_width = int(math.ceil(size[1] * self.__scale[1]))
        return scaled_height, scaled_width

    def result_size(self, size: tuple[int, int]):
        scaled_height, scaled_width = self.result_size_before_rotation(size)

        if self.__rotate == 0 or self.__rotate == 180:
            return scaled_height, scaled_width
        else:
            return scaled_width, scaled_height

    def create_operator(self, image_shape):
        if self.__rel_camera_matrix is not None and self.__distortion_coefficients is not None:
            in_h = image_shape[0]
            in_w = image_shape[1]
            out_size_before_rotation = self.result_size_before_rotation(image_shape)
            out_h_before_rotation = out_size_before_rotation[0]
            out_w_before_rotation = out_size_before_rotation[1]

            in_res_matrix = np.array([[in_w, 0, 0], [0, in_h, 0], [0, 0, 1]])
            in_abs_camera_matrix = np.matmul(in_res_matrix, self.__rel_camera_matrix)

            out_res_matrix = np.array([[out_w_before_rotation, 0, 0], [0, out_h_before_rotation, 0], [0, 0, 1]])
            out_abs_camera_matrix = np.matmul(out_res_matrix, self.__rel_camera_matrix)

            map_x, map_y = cv2.initUndistortRectifyMap(
                in_abs_camera_matrix,
                self.__distortion_coefficients,
                None,
                out_abs_camera_matrix,
                (out_w_before_rotation, out_h_before_rotation),
                cv2.CV_32FC1,
            )

            # FIXME: Order of the operations rotate, flip_h, flip_v, scale:
            #        - apply scale first ((w, h) are wrt the camera image size)
            #        - then apply flips (to correct mirrored camera images)
            #        - then apply rotation (this is to orient the image correctly wrt a canonical coordinate system)
            # TODO: Document order in the manual

            # flipping and rotations are done on the remap matrices to avoid resampling the image multiple times

            # flip horizontally
            if self.__flip_h:
                map_x = np.fliplr(map_x)
                map_y = np.fliplr(map_y)

            # flip vertically
            if self.__flip_v:
                map_x = np.flipud(map_x)
                map_y = np.flipud(map_y)

            # rotate
            map_x = np.rot90(map_x, k=self.__rotate // 90)
            map_y = np.rot90(map_y, k=self.__rotate // 90)

            # if one axis is scaled up and the other down, use cv2.INTER_AREA as a compromise
            # (equivalent to cv2.INTER_NEAREST for the upscaled axis)
            scale = self.__scale
            interpolation_flag = cv2.INTER_LINEAR if scale[0] >= 1.0 and scale[1] >= 1.0 else cv2.INTER_AREA

            return lambda img: cv2.remap(img, map_x, map_y, interpolation_flag)
        else:
            return lambda img: img

    def __call__(self, image):
        if self.last_image_shape != image.shape or self.operator is None:
            self.operator = self.create_operator(image.shape)
            self.last_image_shape = image.shape

        return self.operator(image)

    @staticmethod
    def create_from_config(config):
        camera_config = config["camera"]

        calibration_config = camera_config["calibration"]
        rel_camera_matrix, distortion_coefficients = (
            np.array(calibration_config["matrix"]),
            np.array(calibration_config["distortion"]),
        )
        scale, rotate, flip_h, flip_v = (
            camera_config["scale"],
            camera_config["rotate"],
            camera_config["flipH"],
            camera_config["flipV"],
        )
        return Preprocess(rel_camera_matrix, distortion_coefficients, scale, rotate, flip_h, flip_v)
