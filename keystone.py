from collections import namedtuple

import cv2
import numpy as np

from utils import load_coefficients
from tags import TAGS
from http_json_poster import HttpJsonPoster
from frame import detect_frame_corners
from roi import compute_roi_size, compute_roi_matrix, compute_roi_points
from tag_detector import TagDetector, tiles_to_image


BLOCK_SIZE = 4
GRID_SHAPE = (16, 16)
GRID_SIZE = (GRID_SHAPE[0] * BLOCK_SIZE, GRID_SHAPE[1] * BLOCK_SIZE)

frame_width = 740
frame_height = 740
frame_points = np.array(
    [[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]],
    dtype=np.float32,
)

margin_trbl = (20, 20, 20, 20)

gaps = (4, 4)

roiWidth = frame_width - (margin_trbl[1] + margin_trbl[3])
roiHeight = frame_height - (margin_trbl[0] + margin_trbl[2])

roiPoints = np.array(
    [
        [margin_trbl[1], margin_trbl[0]],
        [frame_width - margin_trbl[3], margin_trbl[0]],
        [frame_width - margin_trbl[3], frame_height - margin_trbl[2]],
        [margin_trbl[1], frame_height - margin_trbl[2]],
    ],
    dtype=np.float32,
)

roiToFrameH = np.matmul(
    np.array(
        [[roiWidth / frame_width, 0, 0], [0, roiHeight / frame_height, 0], [0, 0, 1]]
    ),
    cv2.findHomography(roiPoints, frame_points, cv2.LMEDS)[0],
)

gridToRoiScale = np.array(
    [[GRID_SIZE[0] / roiWidth, 0, 0], [0, GRID_SIZE[1] / roiHeight, 0], [0, 0, 1]]
)
gridMove = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
gridToFrameH = np.matmul(gridMove, np.matmul(gridToRoiScale, roiToFrameH))

lastH = gridToFrameH

print(frame_points)
print(roiPoints)
print(lastH)


def undistort(img, camera_matrix, distortion_coefficients):
    return cv2.undistort(img, camera_matrix, distortion_coefficients, None, None)


def compute_roi(undistorted_img_gray):
    frame = detect_frame_corners(undistorted_img_gray)

    if frame is not None:
        real_frame_size = (frame_height, frame_width)
        roi_size = compute_roi_size(real_frame_size, margin_trbl, frame.corners)
        roi_matrix = compute_roi_matrix(
            real_frame_size, margin_trbl, frame.corners, roi_size
        )

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


def visualize(img, camera_matrix, distortion_coefficients, intermediates):
    undistorted_img = undistort(img, camera_matrix, distortion_coefficients)
    if intermediates.roi is not None:
        draw_frame_and_roi(undistorted_img, intermediates.roi)
    cv2.imshow("detected frame and roi", undistorted_img)

    if intermediates.images is not None and intermediates.images.roi is not None:
        cv2.imshow("region of interest", intermediates.images.roi)
    else:
        cv2.destroyWindow("region of interest")

    if intermediates.tiles is not None:
        tiles_img = tiles_to_image(intermediates.tiles, scale_factor=7)
        cv2.imshow("tiles", tiles_img)
    else:
        cv2.destroyWindow("tiles")


def extract_roi_and_detect_tags(
    img, camera_matrix, distortion_coefficients, tag_detector
):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undistorted_gray = undistort(img_gray, camera_matrix, distortion_coefficients)

    roi = compute_roi(undistorted_gray)

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


def from_camera(camera_matrix, distortion_coefficients, tag_detector, http_json_poster):
    capture = cv2.VideoCapture(0)

    last_detected_tags = tag_detector.create_empty_tags()
    while True:
        ret, src = capture.read()
        detected_tags, intermediates = extract_roi_and_detect_tags(
            src, camera_matrix, distortion_coefficients, tag_detector
        )
        if detected_tags is not None:
            if not np.array_equal(last_detected_tags, detected_tags):
                print("new tags:\n", detected_tags)
                last_detected_tags = detected_tags
                http_json_poster.request_post({"cells": last_detected_tags.tolist()})

        visualize(src, camera_matrix, distortion_coefficients, intermediates)
        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def from_file(camera_matrix, distortion_coefficients, tag_detector):
    src = cv2.imread("snapshot.jpg")
    detected_tags, intermediates = extract_roi_and_detect_tags(
        src, camera_matrix, distortion_coefficients, tag_detector
    )
    print("tags", detected_tags)
    visualize(src, camera_matrix, distortion_coefficients, intermediates)
    cv2.waitKey()


if __name__ == "__main__":

    def init():
        camera_matrix, distortion_coefficients = load_coefficients("camera-profile.yml")
        http_json_poster = HttpJsonPoster("http://localhost:4848/city/map")
        tag_detector = TagDetector(
            GRID_SHAPE,
            (BLOCK_SIZE, BLOCK_SIZE),
            (gaps[0] / roiHeight, gaps[1] / roiWidth),
            TAGS,
        )
        from_file(camera_matrix, distortion_coefficients, tag_detector)
        from_camera(
            camera_matrix, distortion_coefficients, tag_detector, http_json_poster
        )

    init()
