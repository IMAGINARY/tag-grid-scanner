import json
import math
import sys
import jsonpointer
import threading
import time

import cv2
import numpy as np

from taggridscanner.notification_manager import NotificationManager
from taggridscanner.utils import (
    compute_rel_gap,
    create_preprocessor,
    setup_video_capture,
    create_scan_result_transformer,
    create_roi_detector,
    create_frame_reader,
)
from taggridscanner.http_json_poster import HttpJsonPoster
from taggridscanner.tag_detector import TagDetector, tiles_to_image


def extract_roi(undistorted_img, roi_matrix, roi_shape):
    roi_size = roi_shape[::-1]
    return cv2.warpPerspective(
        undistorted_img, roi_matrix, roi_size, flags=cv2.INTER_AREA
    )


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
                img.shape[0],
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


def visualize_roi(preprocessed, roi_visualizers):
    preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    for roi_visualizer in roi_visualizers:
        preprocessed = roi_visualizer(preprocessed)
    roi_viz_window_name = "ROI on undistorted input"
    #    cv2.namedWindow(roi_viz_window_name, cv2.WINDOW_NORMAL)
    #    cv2.resizeWindow(roi_viz_window_name, preprocessed.shape[1], preprocessed.shape[0])
    cv2.imshow(roi_viz_window_name, preprocessed)


def visualize_grid(window_name, img, tag_detector):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_grid(
        img,
        tag_detector.grid_shape,
        tag_detector.tag_shape,
        tag_detector.rel_gaps,
    )
    #    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #    cv2.resizeWindow(window_name, img.shape[1], img.shape[0])
    cv2.imshow(window_name, img)


def visualize_tiles(tiles, tag_detector):
    tiles_img = tiles_to_image(tiles, scale_factor=8)
    visualize_grid("Tiles", tiles_img, tag_detector)


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

    visualizers = [
        lambda: visualize_grid("ROI extracted", roi_img, tag_detector),
        lambda: visualize_grid(
            "ROI extracted, threshold applied", roi_img_threshold, tag_detector
        ),
        lambda: visualize_tiles(tiles, tag_detector),
    ]

    return detected_tags, visualizers


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
    detect_roi,
    tag_detector,
    notify,
):
    read_frame = create_frame_reader(capture)
    is_first_frame = True
    last_detected_tags = tag_detector.create_empty_tags()
    roi = None
    roi_visualizers = []
    img_to_renew_roi = None
    img_to_renew_roi_cond = threading.Condition()

    def renew_roi():
        nonlocal img_to_renew_roi
        while True:
            if img_to_renew_roi is not None:
                new_roi, new_roi_visualizers = detect_roi(img_to_renew_roi)
                if new_roi is not None:
                    nonlocal roi
                    nonlocal roi_visualizers
                    roi = new_roi
                    roi_visualizers = new_roi_visualizers
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

        src = read_frame()

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
                if is_first_frame:
                    # first frame: compute immediately in same thread
                    roi, roi_visualizers = detect_roi(undistorted_gray)
                    is_first_frame = False
                else:
                    # other frames: compute in background thread
                    with img_to_renew_roi_cond:
                        img_to_renew_roi = undistorted_gray
                        img_to_renew_roi_cond.notifyAll()

            if src_has_changed:
                if roi is not None:
                    visualize_roi(undistorted_gray, roi_visualizers)
                    (
                        detected_tags,
                        extracted_roi_visualizers,
                    ) = extract_roi_and_detect_tags(undistorted_gray, roi, tag_detector)
                    if detected_tags is not None:
                        if not np.array_equal(last_detected_tags, detected_tags):
                            notify(detected_tags.tolist())
                            last_detected_tags = detected_tags

                    for extracted_roi_visualizer in extracted_roi_visualizers:
                        extracted_roi_visualizer()
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


def scan(args):
    config_with_defaults = args["config-with-defaults"]
    capture = setup_video_capture(config_with_defaults["camera"])
    preprocess = create_preprocessor(config_with_defaults["camera"])
    notify = create_notifier(config_with_defaults["notify"])

    block_shape = tuple(config_with_defaults["dimensions"]["tile"])
    grid_shape = tuple(config_with_defaults["dimensions"]["grid"])

    abs_frame_size = tuple(config_with_defaults["dimensions"]["size"])
    abs_margin_trbl = tuple(config_with_defaults["dimensions"]["padding"])
    abs_gap = tuple(config_with_defaults["dimensions"]["gap"])
    crop_factors = tuple(config_with_defaults["dimensions"]["crop"])

    detect_roi = create_roi_detector(config_with_defaults["dimensions"])

    rel_gap = compute_rel_gap(abs_frame_size, abs_margin_trbl, abs_gap)

    tag_detector = TagDetector(
        grid_shape, block_shape, rel_gap, config_with_defaults["tags"], crop_factors
    )

    capture_and_detect(capture, preprocess, detect_roi, tag_detector, notify)
