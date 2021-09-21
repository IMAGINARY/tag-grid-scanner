import json
import sys
import jsonpointer
import threading
import time
from collections import namedtuple

import cv2
import numpy as np

from arguments import get_arguments
from config import get_config
from notification_manager import NotificationManager
from utils import load_coefficients
from http_json_poster import HttpJsonPoster
from frame import detect_frame_corners
from roi import compute_roi_shape, compute_roi_matrix, compute_roi_points
from tag_detector import TagDetector, tiles_to_image


def compute_roi(undistorted_img_gray, rel_margin_trbl, aspect_ratio):
    frame = detect_frame_corners(undistorted_img_gray)

    if frame is not None:
        roi_shape = compute_roi_shape(rel_margin_trbl, frame.corners, aspect_ratio)
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
    cv2.drawContours(undistorted_img, frame_contours, -1, (255, 0, 0), 1)
    cv2.polylines(undistorted_img, frame_corners, True, (0, 0, 255), 1)

    roi_corners = np.array(np.expand_dims(roi.corners, axis=0), dtype=np.int32)
    cv2.polylines(undistorted_img, roi_corners, True, (0, 255, 255), 1)


def draw_grid(img, grid_shape, tile_shape):
    rows = grid_shape[0] * tile_shape[0]
    cols = grid_shape[1] * tile_shape[1]
    for x in range(1, cols):
        cv2.line(
            img,
            (int((x * img.shape[1]) / cols), 0),
            (int((x * img.shape[1]) / cols), img.shape[0]),
            (255, 128, 128),
            thickness=1,
        )
    for y in range(1, rows):
        cv2.line(
            img,
            (0, int((y * img.shape[0]) / rows)),
            (img.shape[1], int((y * img.shape[0]) / rows)),
            (255, 128, 128),
            thickness=1,
        )
    for x in range(1, grid_shape[1]):
        cv2.line(
            img,
            (int((x * img.shape[1]) / grid_shape[1]), 0),
            (int((x * img.shape[1]) / grid_shape[1]), img.shape[0]),
            (0, 0, 255),
            thickness=1,
        )
    for y in range(1, grid_shape[0]):
        cv2.line(
            img,
            (0, int((y * img.shape[0]) / grid_shape[0])),
            (img.shape[1], int((y * img.shape[0]) / grid_shape[0])),
            (0, 0, 255),
            thickness=1,
        )


def visualize(
    preprocessed,
    roi,
    roi_image,
    roi_image_threshold,
    tiles,
):
    if roi is not None:
        preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        draw_frame_and_roi(preprocessed, roi)
    cv2.namedWindow("detected frame and roi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "detected frame and roi", preprocessed.shape[1], preprocessed.shape[1]
    )
    cv2.imshow("detected frame and roi", preprocessed)

    if roi_image is not None:
        roi_image_bgr = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        draw_grid(roi_image_bgr, (16, 16), (4, 4))
        cv2.imshow("region of interest", roi_image_bgr)
    else:
        cv2.destroyWindow("region of interest")

    if roi_image_threshold is not None:
        roi_image_threshold_bgr = cv2.cvtColor(roi_image_threshold, cv2.COLOR_GRAY2BGR)
        draw_grid(roi_image_threshold_bgr, (16, 16), (4, 4))
        cv2.imshow("region of interest (threshold)", roi_image_threshold_bgr)
    else:
        cv2.destroyWindow("region of interest")

    if tiles is not None:
        tiles_img = tiles_to_image(tiles, scale_factor=8)
        tiles_img_bgr = cv2.cvtColor(tiles_img, cv2.COLOR_GRAY2BGR)
        draw_grid(tiles_img_bgr, (16, 16), (4, 4))
        cv2.imshow("tiles", tiles_img_bgr)
    else:
        cv2.destroyWindow("tiles")


def extract_roi_and_detect_tags(undistorted_img_gray, roi, tag_detector):
    roi_img = extract_roi(undistorted_img_gray, roi.matrix, roi.shape)
    roi_img_threshold = cv2.adaptiveThreshold(
        roi_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 10
    )

    tiles = tag_detector.extract_tiles(roi_img_threshold)
    detected_tags = tag_detector.detect_tags(tiles)

    Intermediates = namedtuple(
        "Intermediates", ["roi_image", "roi_image_threshold", "tiles"]
    )

    return detected_tags, Intermediates(
        roi_image=roi_img,
        roi_image_threshold=roi_img_threshold,
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


def create_notifier(notify_config):
    notifiers = []
    if notify_config["stdout"]:
        notifiers.append(lambda s: print(s, file=sys.stdout))
    if notify_config["stderr"]:
        notifiers.append(lambda s: print(s, file=sys.stderr))
    if notify_config["remote"]:
        http_json_poster = HttpJsonPoster(notify_config["url"])
        notifiers.append(lambda s: http_json_poster.request_post(s))

    notification_manager = NotificationManager(
        notifiers, notify_config["interval"] if notify_config["repeat"] else None
    )

    template = notify_config["template"]
    assign_to = notify_config["assignTo"]

    def notify(new_tags):
        notification_obj = jsonpointer.set_pointer(template, assign_to, new_tags, False)
        notification = json.dumps(notification_obj)
        notification_manager.notify(notification)

    return notify


def capture_and_detect(
    capture,
    preprocess,
    rel_margin_trbl,
    tag_detector,
    notify,
):
    first_frame_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
    last_detected_tags = tag_detector.create_empty_tags()
    roi = None
    img_to_renew_roi = None
    img_to_renew_roi_cond = threading.Condition()
    aspect_ratio = (tag_detector.grid_shape[0] * tag_detector.tag_shape[0]) / (
        tag_detector.grid_shape[1] * tag_detector.tag_shape[1]
    )

    def renew_roi():
        nonlocal img_to_renew_roi
        while True:
            if img_to_renew_roi is not None:
                new_roi = compute_roi(img_to_renew_roi, rel_margin_trbl, aspect_ratio)
                if new_roi is not None:
                    nonlocal roi
                    roi = new_roi
            with img_to_renew_roi_cond:
                img_to_renew_roi_cond.wait()

    roi_thread = threading.Thread(target=renew_roi, daemon=True)
    roi_thread.start()

    renew_roi_interval = 1
    renew_roi_ts = float("-inf")
    last_src_gray = None
    src_has_changed = True
    src_for_roi_has_changed = True

    start_ts = time.perf_counter()

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
            src_has_changed = True  # ret > 10.0
            src_for_roi_has_changed = src_has_changed
        last_src_gray = src_gray

        if src_for_roi_has_changed:
            undistorted_gray = preprocess(src_gray)
            ts = time.perf_counter()
            if ts > renew_roi_ts + renew_roi_interval and start_ts + 5.0 >= ts:
                renew_roi_ts = ts
                if capture.get(cv2.CAP_PROP_POS_FRAMES) == first_frame_index + 1.0:
                    # first frame: compute immediately in same thread
                    roi = compute_roi(undistorted_gray, rel_margin_trbl, aspect_ratio)
                else:
                    # other frames: compute in background thread
                    with img_to_renew_roi_cond:
                        img_to_renew_roi = undistorted_gray
                        img_to_renew_roi_cond.notifyAll()

            if src_has_changed:
                intermediates = None
                if roi is not None:
                    detected_tags, intermediates = extract_roi_and_detect_tags(
                        undistorted_gray, roi, tag_detector
                    )

                    if detected_tags is not None:
                        if not np.array_equal(last_detected_tags, detected_tags):
                            notify(detected_tags.tolist())

                visualize(
                    undistorted_gray,
                    roi,
                    intermediates.roi_image if intermediates is not None else None,
                    intermediates.roi_image_threshold
                    if intermediates is not None
                    else None,
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
        notify = create_notifier(config_with_defaults["notify"])

        block_shape = tuple(config_with_defaults["dimensions"]["tile"])
        grid_shape = tuple(config_with_defaults["dimensions"]["grid"])

        abs_frame_size = tuple(config_with_defaults["dimensions"]["size"])
        abs_margin_trbl = tuple(config_with_defaults["dimensions"]["padding"])
        abs_gap_hv = tuple(config_with_defaults["dimensions"]["gap"])

        rel_gap_hv = compute_rel_gap_hv(abs_frame_size, abs_margin_trbl, abs_gap_hv)
        rel_gap_vh = rel_gap_hv[::-1]

        rel_margin_trbl = compute_rel_margin_trbl(abs_frame_size, abs_margin_trbl)
        mirror_tags = bool(config_with_defaults["camera"]["flipH"]) != bool(
            config_with_defaults["camera"]["flipV"]
        )
        tag_detector = TagDetector(
            grid_shape,
            block_shape,
            rel_gap_vh,
            config_with_defaults["tags"],
            mirror_tags,
        )

        capture_and_detect(capture, preprocess, rel_margin_trbl, tag_detector, notify)

    init()
