from collections import namedtuple

import cv2
import numpy as np

from utils import load_coefficients
from tags import TAGS
from http_json_poster import HttpJsonPoster
from frame import detect_frame_corners
from roi import compute_roi_size, compute_roi_matrix, compute_roi_points
from tag_detector import TagDetector, tiles_to_image


def undistort(img, camera_matrix, distortion_coefficients):
    return cv2.undistort(img, camera_matrix, distortion_coefficients, None, None)


def compute_roi(undistorted_img_gray, rel_margin_trbl):
    frame = detect_frame_corners(undistorted_img_gray)

    if frame is not None:
        roi_size = compute_roi_size(rel_margin_trbl, frame.corners)
        roi_matrix = compute_roi_matrix(rel_margin_trbl, frame.corners, roi_size)

        Roi = namedtuple("Roi", ["size", "matrix", "corners", "frame"])

        return Roi(
            size=roi_size,
            matrix=roi_matrix,
            corners=compute_roi_points(roi_size, roi_matrix),
            frame=frame,
        )
    else:
        return None


def extract_roi(undistorted_img, roi_matrix, roi_size):
    return cv2.warpPerspective(
        undistorted_img, roi_matrix, roi_size, flags=cv2.INTER_AREA
    )


def draw_frame_and_roi(undistorted_img, roi):
    frame = roi.frame
    frame_corners = np.moveaxis(frame.corners, source=1, destination=0)
    frame_contours = np.expand_dims(frame.contour, axis=0)
    cv2.drawContours(undistorted_img, frame_contours, -1, (0, 255, 255), 4)
    cv2.polylines(undistorted_img, frame_corners, True, (0, 0, 255), 2)

    roi_corners = np.array(np.expand_dims(roi.corners, axis=0), dtype=np.int32)
    cv2.polylines(undistorted_img, roi_corners, True, (255, 0, 0), 2)


def visualize(
    img,
    camera_matrix,
    distortion_coefficients,
    roi,
    roi_image,
    tiles,
):
    undistorted_img = undistort(img, camera_matrix, distortion_coefficients)
    if roi is not None:
        draw_frame_and_roi(undistorted_img, roi)
    cv2.imshow("detected frame and roi", undistorted_img)

    if roi_image is not None:
        cv2.imshow("region of interest", roi_image)
    else:
        cv2.destroyWindow("region of interest")

    if tiles is not None:
        tiles_img = tiles_to_image(tiles, scale_factor=7)
        cv2.imshow("tiles", tiles_img)
    else:
        cv2.destroyWindow("tiles")


def extract_roi_and_detect_tags_old(
    img, camera_matrix, distortion_coefficients, rel_margin_trbl, tag_detector
):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undistorted_gray = undistort(img_gray, camera_matrix, distortion_coefficients)

    roi = compute_roi(undistorted_gray, rel_margin_trbl)

    if roi is not None:
        roi_img = extract_roi(undistorted_gray, roi.matrix, roi.size)
        tiles = tag_detector.extract_tiles(roi_img)
        detected_tags = tag_detector.detect_tags(tiles)
    else:
        roi_img = None
        tiles = None
        detected_tags = None

    Images = namedtuple("Images", ["gray", "undistorted_gray", "roi"])
    images = Images(gray=img_gray, undistorted_gray=undistorted_gray, roi=roi_img)

    Intermediates = namedtuple("Intermediates", ["images", "roi", "tiles"])

    return detected_tags, Intermediates(
        images=images,
        roi=roi,
        tiles=tiles,
    )


def extract_roi_and_detect_tags(undistorted_img_gray, roi, tag_detector):
    roi_img = extract_roi(undistorted_img_gray, roi.matrix, roi.size)
    tiles = tag_detector.extract_tiles(roi_img)
    detected_tags = tag_detector.detect_tags(tiles)

    Intermediates = namedtuple("Intermediates", ["roi_image", "tiles"])

    return detected_tags, Intermediates(
        roi_image=roi_img,
        tiles=tiles,
    )


def from_camera(
    camera_matrix,
    distortion_coefficients,
    rel_margin_trbl,
    tag_detector,
    http_json_poster,
):
    capture = cv2.VideoCapture(0)

    last_detected_tags = tag_detector.create_empty_tags()
    while True:
        ret, src = capture.read()
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        undistorted_gray = undistort(src_gray, camera_matrix, distortion_coefficients)

        roi = compute_roi(undistorted_gray, rel_margin_trbl)

        if roi is not None:
            detected_tags, intermediates = extract_roi_and_detect_tags(
                undistorted_gray, roi, tag_detector
            )
            if detected_tags is not None:
                if not np.array_equal(last_detected_tags, detected_tags):
                    print("new tags:\n", detected_tags)
                    last_detected_tags = detected_tags
                    http_json_poster.request_post(
                        {"cells": last_detected_tags.tolist()}
                    )

        visualize(
            src,
            camera_matrix,
            distortion_coefficients,
            roi,
            intermediates.roi_image,
            intermediates.tiles,
        )
        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def from_file(camera_matrix, distortion_coefficients, rel_margin_trbl, tag_detector):
    src = cv2.imread("snapshot.jpg")
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    undistorted_gray = undistort(src_gray, camera_matrix, distortion_coefficients)

    roi = compute_roi(undistorted_gray, rel_margin_trbl)
    if roi is not None:
        detected_tags, intermediates = extract_roi_and_detect_tags(
            undistorted_gray, roi, tag_detector
        )
        print("tags", detected_tags)
    visualize(
        src,
        camera_matrix,
        distortion_coefficients,
        roi,
        intermediates.roi_image,
        intermediates.tiles,
    )
    cv2.waitKey()


def compute_abs_roi_size(frame_size, margin_trbl):
    roi_height = frame_size[0] - (margin_trbl[0] + margin_trbl[2])
    roi_width = frame_size[1] - (margin_trbl[1] + margin_trbl[3])
    return roi_height, roi_width


def compute_rel_roi_size(frame_size, margin_trbl):
    abs_roi_size = compute_abs_roi_size(frame_size, margin_trbl)
    return abs_roi_size[0] / frame_size[0], abs_roi_size[0] / frame_size[0]


def compute_rel_margin_trbl(frame_size, margin_trbl):
    return (
        margin_trbl[0] / frame_size[0],
        margin_trbl[1] / frame_size[1],
        margin_trbl[2] / frame_size[0],
        margin_trbl[3] / frame_size[1],
    )


def compute_rel_gap_size(frame_size, margin_trbl, gaps):
    abs_roi_size = compute_abs_roi_size(frame_size, margin_trbl)
    return gaps[0] / abs_roi_size[0], gaps[1] / abs_roi_size[1]


if __name__ == "__main__":

    def init():
        camera_matrix, distortion_coefficients = load_coefficients("camera-profile.yml")
        http_json_poster = HttpJsonPoster("http://localhost:4848/city/map")

        block_size = 4
        grid_shape = (16, 16)

        abs_frame_size = (740, 740)
        abs_margin_trbl = (20, 20, 20, 20)
        abs_gap_size = (4, 4)

        rel_gap_size = compute_rel_gap_size(
            abs_frame_size, abs_margin_trbl, abs_gap_size
        )
        rel_margin_trbl = compute_rel_margin_trbl(abs_frame_size, abs_margin_trbl)
        tag_detector = TagDetector(
            grid_shape,
            (block_size, block_size),
            rel_gap_size,
            TAGS,
        )

        from_file(camera_matrix, distortion_coefficients, rel_margin_trbl, tag_detector)
        from_camera(
            camera_matrix,
            distortion_coefficients,
            rel_margin_trbl,
            tag_detector,
            http_json_poster,
        )

    init()
