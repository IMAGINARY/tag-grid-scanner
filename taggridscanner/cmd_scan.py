import json
import math
import sys
import jsonpointer
import threading
import time
from collections import namedtuple

import cv2
import numpy as np

from .notification_manager import NotificationManager
from .utils import (
    load_coefficients,
    compute_rel_margin_trbl,
    compute_rel_gap,
    create_preprocessor,
    compute_rel_margin_trbl,
    setup_video_capture,
    create_scan_result_transformer,
)
from .http_json_poster import HttpJsonPoster
from .frame import detect_frame_corners
from .roi import (
    compute_roi_shape,
    compute_roi_matrix,
    compute_roi_points,
    compute_roi_aspect_ratio,
)
from .tag_detector import TagDetector, tiles_to_image


def compute_roi(undistorted_img_gray, rel_margin_trbl, roi_aspect_ratio):
    frame = detect_frame_corners(undistorted_img_gray)

    if frame is not None:
        roi_shape = compute_roi_shape(rel_margin_trbl, frame.corners, roi_aspect_ratio)
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


def draw_grid(img, grid_shape, tag_shape, rel_gap=(0.0, 0.0)):
    padded_img_shape = (
        img.shape[0] * (1 + rel_gap[0]),
        img.shape[1] * (1 + rel_gap[1]),
    )
    tile_size_with_gap = (
        padded_img_shape[0] / grid_shape[0],
        padded_img_shape[1] / grid_shape[1],
    )
    gap_size = (img.shape[0] * rel_gap[0], img.shape[1] * rel_gap[1])
    tile_size = (
        tile_size_with_gap[0] - gap_size[0],
        tile_size_with_gap[1] - gap_size[1],
    )

    for y_grid in range(0, grid_shape[0]):
        row_start = math.floor(tile_size_with_gap[0] * y_grid)
        row_end = math.ceil(row_start + tile_size[0])
        row = img[row_start:row_end, :]
        for y_tile in range(1, tag_shape[0]):
            y = int((y_tile * row.shape[0]) / tag_shape[0])
            cv2.line(
                row,
                (0, y),
                (row.shape[1], y),
                (255, 128, 128),
                thickness=1,
            )

    for x_grid in range(0, grid_shape[1]):
        col_start = math.floor(tile_size_with_gap[1] * x_grid)
        col_end = math.ceil(col_start + tile_size[1])
        col = img[:, col_start:col_end]
        for x_tile in range(1, tag_shape[1]):
            x = int((x_tile * col.shape[1]) / tag_shape[1])
            cv2.line(
                col,
                (x, 0),
                (x, col.shape[0]),
                (255, 128, 128),
                thickness=1,
            )

    h_line_width = max(1, int(rel_gap[1] * img.shape[1]))
    for y in range(1, grid_shape[1]):
        cv2.line(
            img,
            (int((y * padded_img_shape[1]) / grid_shape[1] - h_line_width / 2.0), 0),
            (
                int((y * padded_img_shape[1]) / grid_shape[1] - h_line_width / 2),
                img.shape[1],
            ),
            (0, 0, 255),
            thickness=h_line_width,
        )
    v_line_width = max(1, int(rel_gap[0] * img.shape[0]))
    for x in range(1, grid_shape[0]):
        cv2.line(
            img,
            (0, int((x * padded_img_shape[0]) / grid_shape[0] - v_line_width / 2.0)),
            (
                img.shape[1],
                int((x * padded_img_shape[0]) / grid_shape[0] - v_line_width / 2.0),
            ),
            (0, 0, 255),
            thickness=v_line_width,
        )


def visualize(preprocessed, roi, roi_image, roi_image_threshold, tiles, tag_detector):
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
        draw_grid(
            roi_image_bgr,
            tag_detector.grid_shape,
            tag_detector.tag_shape,
            tag_detector.rel_gaps,
        )
        cv2.imshow("region of interest", roi_image_bgr)
    else:
        cv2.destroyWindow("region of interest")

    if roi_image_threshold is not None:
        roi_image_threshold_bgr = cv2.cvtColor(roi_image_threshold, cv2.COLOR_GRAY2BGR)
        draw_grid(
            roi_image_threshold_bgr,
            tag_detector.grid_shape,
            tag_detector.tag_shape,
            tag_detector.rel_gaps,
        )
        cv2.imshow("region of interest (threshold)", roi_image_threshold_bgr)
    else:
        cv2.destroyWindow("region of interest")

    if tiles is not None:
        tiles_img = tiles_to_image(tiles, scale_factor=8)
        tiles_img_bgr = cv2.cvtColor(tiles_img, cv2.COLOR_GRAY2BGR)
        draw_grid(tiles_img_bgr, tag_detector.grid_shape, tag_detector.tag_shape)
        cv2.imshow("tiles", tiles_img_bgr)
    else:
        cv2.destroyWindow("tiles")


def extract_roi_and_detect_tags(undistorted_img_gray, roi, tag_detector):
    roi_img = extract_roi(undistorted_img_gray, roi.matrix, roi.shape)
    thresholding_block_size = max(
        3,
        int(
            (roi_img.shape[0] + roi_img.shape[1])
            / (tag_detector.grid_shape[0] + tag_detector.grid_shape[1])
        ),
    )
    thresholding_block_size += 1 - thresholding_block_size % 2  # must be odd
    roi_img_threshold = cv2.adaptiveThreshold(
        roi_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        thresholding_block_size,
        10,
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

    scan_result_transformer = create_scan_result_transformer(notify_config)

    def notify(new_tags):
        new_tags = scan_result_transformer(new_tags)
        notification_obj = jsonpointer.set_pointer(template, assign_to, new_tags, False)
        notification = json.dumps(notification_obj)
        notification_manager.notify(notification)

    return notify


def capture_and_detect(
    capture,
    preprocess,
    rel_margin_trbl,
    roi_aspect_ratio,
    tag_detector,
    notify,
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
                new_roi = compute_roi(
                    img_to_renew_roi, rel_margin_trbl, roi_aspect_ratio
                )
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
    while True:
        frame_start_ts = time.perf_counter()

        ret, src = capture.read()

        if not ret:
            # reach end of stream -> rewind
            # (can also happen when there is an input error due,
            # but there is no way in OpenCV to tell the difference)
            # maybe switch to PyAV for capturing
            capture.set(cv2.CAP_PROP_POS_MSEC, 0.0)
            continue

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
                    roi = compute_roi(
                        undistorted_gray, rel_margin_trbl, roi_aspect_ratio
                    )
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
                            last_detected_tags = detected_tags

                visualize(
                    undistorted_gray,
                    roi,
                    intermediates.roi_image if intermediates is not None else None,
                    intermediates.roi_image_threshold
                    if intermediates is not None
                    else None,
                    intermediates.tiles if intermediates is not None else None,
                    tag_detector,
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


def scan(args, config, config_with_defaults):
    capture = setup_video_capture(config_with_defaults["camera"])
    preprocess = create_preprocessor(config_with_defaults["camera"])
    notify = create_notifier(config_with_defaults["notify"])

    block_shape = tuple(config_with_defaults["dimensions"]["tile"])
    grid_shape = tuple(config_with_defaults["dimensions"]["grid"])

    abs_frame_size = tuple(config_with_defaults["dimensions"]["size"])
    abs_margin_trbl = tuple(config_with_defaults["dimensions"]["padding"])
    abs_gap = tuple(config_with_defaults["dimensions"]["gap"])
    crop_factors = tuple(config_with_defaults["dimensions"]["crop"])
    roi_aspect_ratio = compute_roi_aspect_ratio(abs_frame_size, abs_margin_trbl)

    rel_gap = compute_rel_gap(abs_frame_size, abs_margin_trbl, abs_gap)

    rel_margin_trbl = compute_rel_margin_trbl(abs_frame_size, abs_margin_trbl)
    tag_detector = TagDetector(
        grid_shape, block_shape, rel_gap, config_with_defaults["tags"], crop_factors
    )

    capture_and_detect(
        capture, preprocess, rel_margin_trbl, roi_aspect_ratio, tag_detector, notify
    )
