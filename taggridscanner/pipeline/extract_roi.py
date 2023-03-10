import cv2
import math
import numpy as np

from taggridscanner.aux.utils import Functor, rel_corners_to_abs_corners


def create_frame_corners(size):
    width, height = size
    return np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float32,
    )


def create_unit_frame_corners():
    return create_frame_corners((1, 1))


def distance(p0, p1):
    return np.linalg.norm(p1 - p0)


def compute_roi_shape(roi_corners, roi_aspect_ratio):
    p = roi_corners

    dist_v_left = distance(p[3], p[0])
    dist_v_right = distance(p[1], p[2])
    dist_v = max(dist_v_left, dist_v_right)

    dist_h_top = distance(p[0], p[1])
    dist_h_bottom = distance(p[2], p[3])
    dist_h = max(dist_h_top, dist_h_bottom)

    if dist_v * roi_aspect_ratio * dist_v > dist_h * dist_h / roi_aspect_ratio:
        h = math.ceil(dist_v)
        w = math.ceil(dist_v * roi_aspect_ratio)
    else:
        h = math.ceil(dist_h / roi_aspect_ratio)
        w = math.ceil(dist_h)

    return h, w


# transforms the frame into the 1x1 square with top-left corner (0,0)
def to_frame_1x1_mat(detected_frame_corners):
    unit_frame_corners = create_unit_frame_corners()
    h = cv2.findHomography(detected_frame_corners, unit_frame_corners)
    return h[0]


def compute_roi_matrix(image_shape, rel_corners, roi_shape):
    img_height, img_width = image_shape[0:2]
    downscale_mat = np.array(
        [
            [1.0 / img_width, 0, 0],
            [0, 1.0 / img_height, 0],
            [0, 0, 1],
        ]
    )

    roi_height, roi_width = roi_shape
    upscale_mat = np.array(
        [
            [roi_width, 0, 0],
            [0, roi_height, 0],
            [0, 0, 1],
        ]
    )

    unit_roi_mat = to_frame_1x1_mat(rel_corners)

    mat = np.matmul(upscale_mat, np.matmul(unit_roi_mat, downscale_mat))

    return mat


class ExtractROI(Functor):
    def __init__(self, target_aspect_ratio, rel_corners=create_unit_frame_corners()):
        super().__init__()
        self.target_aspect_ratio = target_aspect_ratio
        self.rel_corners = rel_corners

    def __call__(self, image, marker_homography_matrix):
        # compute target ROI size
        abs_corners = rel_corners_to_abs_corners(self.rel_corners, image.shape)
        target_size = compute_roi_shape(abs_corners, self.target_aspect_ratio)

        # compute homography matrix
        mtx = compute_roi_matrix(image.shape, self.rel_corners, target_size)

        # incorporate marker homography matrix
        mtx = np.matmul(mtx, marker_homography_matrix)

        return cv2.warpPerspective(image, mtx, target_size, flags=cv2.INTER_AREA)
