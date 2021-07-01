from collections import namedtuple

import cv2
import numpy as np

Frame = namedtuple("Frame", ["corners", "contour"])


def detect_frame_corners(img, min_rel_area=0.001, max_rel_area=0.999, rel_epsilon=0.1):
    bilaterally_filtered_img = cv2.bilateralFilter(img, 5, 175, 175)

    # finds edges of bilaterally filtered image and displays it
    edge_img = cv2.Canny(bilaterally_filtered_img, 75, 200)

    kernel = np.ones((3, 3), np.uint8)
    edge_img = cv2.dilate(edge_img, kernel)
    edge_img = cv2.erode(edge_img, kernel)

    contours, hierarchy = cv2.findContours(
        edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = edge_img.shape[0] * edge_img.shape[1]

    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if min_rel_area * img_area <= contour_area <= max_rel_area * img_area:
            # find the perimeter of the first closed contour
            perimeter = cv2.arcLength(contour, True)
            # setting the precision
            epsilon = rel_epsilon * perimeter
            # approximating the contour with a polygon
            approx_corners = cv2.approxPolyDP(contour, epsilon, True)
            # check how many vertices has the approximate polygon
            approx_corners_number = len(approx_corners)

            if approx_corners_number == 4 and is_convex(approx_corners):
                # it is a convex quadrilateral
                if (
                    orientation(approx_corners[0], approx_corners[1], approx_corners[2])
                    < 0
                ):
                    # points are CCW, but we need CW
                    approx_corners = np.flip(approx_corners, axis=0)

                top_left = min(approx_corners, key=np.linalg.norm)
                while not np.array_equal(approx_corners[0], top_left):
                    approx_corners = np.roll(approx_corners, 1)

                return Frame(corners=approx_corners, contour=contour)

    return None


def is_convex(points):
    if len(points) < 3:
        return True
    else:  # len >= 3
        last_o = 0
        for i in range(-1, len(points) - 2):
            o = orientation(points[i], points[i + 1], points[i + 2])
            if (last_o < 0 and o > 0) or (last_o > 0 and o < 0):
                return False
            if o != 0.0:
                last_o = o
        return True


def orientation(p0, p1, p2):
    return np.cross(p1 - p0, p2 - p1)[0]
