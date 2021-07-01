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


def extract_roi_and_detect_tags(
    img, camera_matrix, distortion_coefficients, tag_detector
):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undistorted_bgr = undistort(img, camera_matrix, distortion_coefficients)
    undistorted_gray = undistort(img_gray, camera_matrix, distortion_coefficients)
    frame = detect_frame_corners(undistorted_gray)

    if frame is not None:
        corners = np.moveaxis(frame.corners, source=1, destination=0)
        contours = np.expand_dims(frame.contour, axis=0)
        cv2.drawContours(undistorted_bgr, contours, -1, (0, 255, 255), 4)
        cv2.polylines(undistorted_bgr, corners, True, (0, 0, 255), 2)

        real_frame_size = (frame_height, frame_width)
        roi_size = compute_roi_size(real_frame_size, margin_trbl, frame.corners)
        roi_matrix = compute_roi_matrix(
            real_frame_size, margin_trbl, frame.corners, roi_size
        )

        roi_img = cv2.warpPerspective(
            undistorted_gray,
            roi_matrix,
            roi_size,
            flags=cv2.INTER_AREA,
        )

        cv2.imshow("roi", roi_img)
        for p in compute_roi_points(roi_size, roi_matrix):
            cv2.circle(undistorted_bgr, tuple(p), 2, (0, 0, 255), -1)

        tiles = tag_detector.extract_tiles(roi_img)

        tiles_img = tiles_to_image(tiles)
        tiles_img_scale_factor = 7
        tiles_img_scaled = cv2.resize(
            tiles_img * 255,
            (
                tiles_img_scale_factor * GRID_SIZE[0],
                tiles_img_scale_factor * GRID_SIZE[1],
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("tiles", tiles_img_scaled)

        detected_tags = tag_detector.detect_tags(tiles)
        print("tags:\n", detected_tags)

    else:
        print("No frame detected")
        detected_tags = tag_detector.create_empty_tags()
    cv2.imshow("frame", undistorted_bgr)

    return detected_tags


def from_camera(camera_matrix, distortion_coefficients, tag_detector, http_json_poster):
    capture = cv2.VideoCapture(0)

    last_detected_tags = tag_detector.create_empty_tags()
    while True:
        ret, src = capture.read()
        detected_tags = extract_roi_and_detect_tags(
            src, camera_matrix, distortion_coefficients, tag_detector
        )
        if not np.array_equal(last_detected_tags, detected_tags):
            print("new tags:\n", detected_tags)
            last_detected_tags = detected_tags
            http_json_poster.request_post({"cells": last_detected_tags.tolist()})
        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def from_file(camera_matrix, distortion_coefficients, tag_detector):
    src = cv2.imread("snapshot.jpg")
    detected_tags = extract_roi_and_detect_tags(
        src, camera_matrix, distortion_coefficients, tag_detector
    )
    print("tags", detected_tags)
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
        # from_file(camera_matrix, distortion_coefficients, tag_detector)
        from_camera(
            camera_matrix, distortion_coefficients, tag_detector, http_json_poster
        )

    init()
