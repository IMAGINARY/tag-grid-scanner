import threading
import time
from collections import namedtuple

import cv2
import numpy as np

from arguments import get_arguments
from config import get_config
from utils import load_coefficients
from http_json_poster import HttpJsonPoster
from frame import detect_frame_corners
from roi import compute_roi_shape, compute_roi_matrix, compute_roi_points
from tag_detector import TagDetector, tiles_to_image


def undistort(img, camera_matrix, distortion_coefficients):
    return cv2.undistort(img, camera_matrix, distortion_coefficients, None, None)


def compute_roi(undistorted_img_gray, rel_margin_trbl):
    frame = detect_frame_corners(undistorted_img_gray)

    if frame is not None:
        roi_shape = compute_roi_shape(rel_margin_trbl, frame.corners)
        roi_matrix = compute_roi_matrix(rel_margin_trbl, frame.corners, roi_shape)

        Roi = namedtuple("Roi", ["shape", "matrix", "corners", "frame"])

        return Roi(
            shape=roi_shape,
            matrix=roi_matrix,
            corners=compute_roi_points(roi_shape, roi_matrix),
            frame=frame,
        )
    else:
        return None


def extract_roi(undistorted_img, roi_matrix, roi_shape):
    roi_size = roi_shape[::-1]
    return cv2.warpPerspective(
        undistorted_img, roi_matrix, roi_size, flags=cv2.INTER_AREA
    )


def draw_frame_and_roi(undistorted_img, roi):
    frame = roi.frame
    frame_corners = np.array(
        np.moveaxis(frame.corners, source=1, destination=0), dtype=np.int32
    )
    frame_contours = np.expand_dims(frame.contour, axis=0)
    cv2.drawContours(undistorted_img, frame_contours, -1, (0, 255, 255), 4)
    cv2.polylines(undistorted_img, frame_corners, True, (0, 0, 255), 2)

    roi_corners = np.array(np.expand_dims(roi.corners, axis=0), dtype=np.int32)
    cv2.polylines(undistorted_img, roi_corners, True, (255, 0, 0), 2)


def visualize(
    preprocessed,
    roi,
    roi_image,
    tiles,
):
    if roi is not None:
        preprocessed = preprocessed.copy()
        draw_frame_and_roi(preprocessed, roi)
    cv2.imshow("detected frame and roi", preprocessed)

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
        roi_img = extract_roi(undistorted_gray, roi.matrix, roi.shape)
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
    roi_img = extract_roi(undistorted_img_gray, roi.matrix, roi.shape)
    tiles = tag_detector.extract_tiles(roi_img)
    detected_tags = tag_detector.detect_tags(tiles)

    Intermediates = namedtuple("Intermediates", ["roi_image", "tiles"])

    return detected_tags, Intermediates(
        roi_image=roi_img,
        tiles=tiles,
    )


def select_capture_source(camera_config):
    if "filename" in camera_config:
        return camera_config["filename"]
    else:
        return camera_config["id"]


def setup_video_capture(camera_config):
    source = select_capture_source(camera_config)
    capture = cv2.VideoCapture(source)

    if "fourcc" in camera_config:
        s = camera_config["fourcc"]
        fourcc = cv2.VideoWriter_fourcc(s[0], s[1], s[2], s[3])
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)

    if "size" in camera_config:
        [width, height] = camera_config["size"]
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if "fps" in camera_config:
        fps = camera_config["fps"]
        capture.set(cv2.CAP_PROP_FPS, fps)

    if "exposure" in camera_config:
        exposure = camera_config["exposure"]
        capture.set(cv2.CAP_PROP_EXPOSURE, exposure)

    return capture


def create_preprocessor(camera_config):
    camera_matrix, distortion_coefficients = (
        load_coefficients(camera_config["calibration"])
        if "calibration" in camera_config
        else (None, None)
    )

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

    rotate_code = get_rotate_code(camera_config["rotate"])
    flip_code = get_flip_code(camera_config["flipH"], camera_config["flipV"])

    def preprocess(img):
        if camera_matrix is not None and distortion_coefficients is not None:
            img = cv2.undistort(img, camera_matrix, distortion_coefficients, None, None)
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)
        if flip_code is not None:
            img = cv2.flip(img, flip_code)
        return img

    return preprocess


def capture_and_detect(
    capture,
    preprocess,
    rel_margin_trbl,
    tag_detector,
    http_json_poster,
):
    first_frame_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
    last_detected_tags = tag_detector.create_empty_tags()
    roi = None
    img_to_renew_roi = None
    img_to_renew_roi_cond = threading.Condition()

    def renew_roi():
        nonlocal img_to_renew_roi
        while True:
            if img_to_renew_roi is not None:
                new_roi = compute_roi(img_to_renew_roi, rel_margin_trbl)
                if new_roi is not None:
                    nonlocal roi
                    roi = new_roi
            with img_to_renew_roi_cond:
                img_to_renew_roi_cond.wait()

    roi_thread = threading.Thread(target=renew_roi, daemon=True)
    roi_thread.start()

    renew_roi_interval = 5
    renew_roi_ts = float("-inf")
    last_src_gray = None
    src_has_changed = True

    wait_for_key = True
    while capture.get(cv2.CAP_PROP_FRAME_COUNT) == 0.0 or capture.get(
        cv2.CAP_PROP_POS_FRAMES
    ) < capture.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_start_ts = time.perf_counter()
        ret, src = capture.read()

        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if last_src_gray is not None:

            def absdiff(img1, img2):
                a = img1 - img2
                b = np.uint8(img1 < img2) * 254 + 1
                a *= b
                return a

            diff = absdiff(last_src_gray, src_gray)
            ret, thres = cv2.threshold(
                diff, None, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            # cv2.imshow("diff", thres)
            src_has_changed = ret > 10.0
        last_src_gray = src_gray

        if src_has_changed:
            undistorted_gray = preprocess(src_gray)
            ts = time.perf_counter()
            if ts > renew_roi_ts + renew_roi_interval:
                renew_roi_ts = ts
                if capture.get(cv2.CAP_PROP_POS_FRAMES) == first_frame_index + 1.0:
                    # first frame: compute immediately in same thread
                    roi = compute_roi(undistorted_gray, rel_margin_trbl)
                else:
                    # other frames: compute in background thread
                    with img_to_renew_roi_cond:
                        img_to_renew_roi = undistorted_gray
                        img_to_renew_roi_cond.notifyAll()

            intermediates = None
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
                undistorted_gray,
                roi,
                intermediates.roi_image if intermediates is not None else None,
                intermediates.tiles if intermediates is not None else None,
            )
        frame_end_ts = time.perf_counter()
        key = cv2.waitKey(
            max(1, int(1000 / 15) - int(1000 * (frame_end_ts - frame_start_ts)))
        )
        if key == 27:
            wait_for_key = False
            break

    if wait_for_key:
        while True:
            key = cv2.waitKey(10)
            if key == 27:
                break

    capture.release()
    cv2.destroyAllWindows()


def compute_abs_roi_size(frame_size, margin_trbl):
    roi_width = frame_size[0] - (margin_trbl[1] + margin_trbl[3])
    roi_height = frame_size[1] - (margin_trbl[0] + margin_trbl[2])
    return roi_width, roi_height


def compute_rel_roi_size(frame_size, margin_trbl):
    abs_roi_size = compute_abs_roi_size(frame_size, margin_trbl)
    return abs_roi_size[0] / frame_size[0], abs_roi_size[0] / frame_size[0]


def compute_rel_margin_trbl(frame_size, margin_trbl):
    return (
        margin_trbl[0] / frame_size[1],
        margin_trbl[1] / frame_size[0],
        margin_trbl[2] / frame_size[1],
        margin_trbl[3] / frame_size[0],
    )


def compute_rel_gap_hv(frame_size, margin_trbl, gaps):
    abs_roi_size = compute_abs_roi_size(frame_size, margin_trbl)
    return gaps[0] / abs_roi_size[0], gaps[1] / abs_roi_size[1]


if __name__ == "__main__":

    def init():
        args = get_arguments()
        config, config_with_defaults = get_config(args["CONFIG_FILE"][0])

        capture = setup_video_capture(config_with_defaults["camera"])
        preprocess = create_preprocessor(config_with_defaults["camera"])
        http_json_poster = (
            HttpJsonPoster("http://localhost:4848/city/map")
            if config["camera"]["calibration"] is not None
            else None
        )

        block_shape = tuple(config_with_defaults["dimensions"]["tile"])
        grid_shape = tuple(config_with_defaults["dimensions"]["grid"])

        abs_frame_size = tuple(config_with_defaults["dimensions"]["size"])
        abs_margin_trbl = tuple(config_with_defaults["dimensions"]["padding"])
        abs_gap_hv = tuple(config_with_defaults["dimensions"]["gap"])

        rel_gap_hv = compute_rel_gap_hv(abs_frame_size, abs_margin_trbl, abs_gap_hv)
        rel_gap_vh = rel_gap_hv[::-1]

        rel_margin_trbl = compute_rel_margin_trbl(abs_frame_size, abs_margin_trbl)
        tag_detector = TagDetector(
            grid_shape,
            block_shape,
            rel_gap_vh,
            config_with_defaults["tags"],
        )

        capture_and_detect(
            capture, preprocess, rel_margin_trbl, tag_detector, http_json_poster
        )

    init()
