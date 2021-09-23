from collections import namedtuple

import cv2
import numpy as np
import math


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


def compute_roi_shape(rel_margins_trbl, frame_corners, aspect_ratio):
    roi_to_frame_ratio = (
        1 - (rel_margins_trbl[0] + rel_margins_trbl[2]),
        1 - (rel_margins_trbl[1] + rel_margins_trbl[3]),
    )

    p = frame_corners

    dist_v_left = distance(p[3], p[0])
    dist_v_right = distance(p[1], p[2])
    dist_v = max(dist_v_left, dist_v_right)

    dist_h_top = distance(p[0], p[1])
    dist_h_bottom = distance(p[2], p[3])
    dist_h = max(dist_h_top, dist_h_bottom)

    if dist_v * aspect_ratio > dist_h:
        h = math.ceil(roi_to_frame_ratio[0] * dist_v)
        w = math.ceil(roi_to_frame_ratio[1] * dist_v / aspect_ratio)
    else:
        h = math.ceil(roi_to_frame_ratio[0] * dist_h * aspect_ratio)
        w = math.ceil(roi_to_frame_ratio[1] * dist_h)

    return h, w


def compute_roi_matrix(rel_margins_trbl, frame_corners, roi_shape):
    roi_height, roi_width = roi_shape
    scale_mat = np.array(
        [
            [roi_width, 0, 0],
            [0, roi_height, 0],
            [0, 0, 1],
        ]
    )

    unit_frame_mat = to_frame_1x1_mat(frame_corners)

    unit_roi_mat = frame_1x1_to_roi_1x1_mat(rel_margins_trbl)

    mat = np.matmul(scale_mat, np.matmul(unit_roi_mat, unit_frame_mat))

    return mat


def to_homogenous(point):
    return np.append(point, 1)


def to_affine(point):
    return point[:-1] / point[-1]


def compute_roi_points(roi_shape, roi_matrix):
    roi_matrix_inv = np.linalg.inv(roi_matrix)
    roi_height, roi_width = roi_shape
    roi_points = create_frame_corners((roi_width, roi_height))
    for i in range(len(roi_points)):
        roi_point_homogenous = to_homogenous(roi_points[i])
        roi_points[i] = to_affine(np.matmul(roi_matrix_inv, roi_point_homogenous))
    return roi_points


# transforms the frame into the 1x1 square with top-left corner (0,0)
def to_frame_1x1_mat(detected_frame_corners):
    unit_frame_corners = create_unit_frame_corners()
    h = cv2.findHomography(detected_frame_corners, unit_frame_corners, cv2.LMEDS)
    return h[0]


def frame_1x1_to_roi_1x1_mat(rel_margins_trbl):
    roi_x = rel_margins_trbl[3]
    roi_y = rel_margins_trbl[0]
    roi_width = 1 - (rel_margins_trbl[1] + rel_margins_trbl[3])
    roi_height = 1 - (rel_margins_trbl[0] + rel_margins_trbl[2])

    move_mat = np.array([[1, 0, -roi_x], [0, 1, -roi_y], [0, 0, 1]])

    scale_mat = np.array(
        [
            [1.0 / roi_width, 0, 0],
            [0, 1.0 / roi_height, 0],
            [0, 0, 1],
        ]
    )

    return np.matmul(scale_mat, move_mat)


def compute_max_size(corners):
    p = corners
    w = math.ceil(max(distance(p[0], p[1]), distance(p[2], p[3])))
    h = math.ceil(max(distance(p[1], p[2]), distance(p[3], p[0])))

    return h, w


def distance(p0, p1):
    return np.linalg.norm(p1 - p0)
