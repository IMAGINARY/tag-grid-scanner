import sys
import time
import cv2
import numpy as np
from collections import namedtuple

from taggridscanner.frame import (
    detect_frame_corners,
    create_frame_contour_visualizer,
    create_frame_visualizer,
)
from taggridscanner.roi import (
    compute_roi_shape_from_frame,
    compute_roi_matrix_from_frame,
    compute_roi_points,
    compute_roi_aspect_ratio,
    create_roi_visualizer,
)


def compute_abs_roi_size(frame_size, margin_trbl):
    roi_width = frame_size[0] - (margin_trbl[1] + margin_trbl[3])
    roi_height = frame_size[1] - (margin_trbl[0] + margin_trbl[2])
    return roi_width, roi_height


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


def compute_rel_gap(frame_size, margin_trbl, gaps):
    abs_roi_size = compute_abs_roi_size(frame_size, margin_trbl)
    return gaps[0] / abs_roi_size[0], gaps[1] / abs_roi_size[1]


Roi = namedtuple("Roi", ["shape", "matrix"])


def compute_roi_from_frame(frame_corners, rel_margin_trbl, roi_aspect_ratio):
    roi_shape = compute_roi_shape_from_frame(
        rel_margin_trbl, frame_corners, roi_aspect_ratio
    )
    roi_matrix = compute_roi_matrix_from_frame(
        rel_margin_trbl, frame_corners, roi_shape
    )

    return Roi(
        shape=roi_shape,
        matrix=roi_matrix,
    )


def abs_corners_to_rel_corners(abs_corners, img_shape):
    return np.apply_along_axis(
        lambda p: [p[0] / img_shape[1], p[1] / img_shape[0]], 1, abs_corners
    )


def rel_corners_to_abs_corners(rel_corners, img_shape):
    return np.apply_along_axis(
        lambda p: [p[0] * img_shape[1], p[1] * img_shape[0]], 1, rel_corners
    )


def create_roi_detector_manual(dimensions_config):
    abs_frame_size = tuple(dimensions_config["size"])
    abs_margin_trbl = tuple(dimensions_config["padding"])
    roi_aspect_ratio = compute_roi_aspect_ratio(abs_frame_size, abs_margin_trbl)

    relative_frame_corners = extract_and_preprocess_roi_config(dimensions_config)

    def detect_roi(img_gray):
        roi = compute_roi_from_frame(
            rel_corners_to_abs_corners(relative_frame_corners, img_gray.shape),
            (0, 0, 0, 0),
            roi_aspect_ratio,
        )
        roi_visualizer = create_roi_visualizer(
            compute_roi_points(roi.shape, roi.matrix)
        )

        return roi, [roi_visualizer]

    return detect_roi


def create_roi_detector_from_frame(dimensions_config):
    abs_frame_size = tuple(dimensions_config["size"])
    abs_margin_trbl = tuple(dimensions_config["padding"])
    roi_aspect_ratio = compute_roi_aspect_ratio(abs_frame_size, abs_margin_trbl)
    rel_margin_trbl = compute_rel_margin_trbl(abs_frame_size, abs_margin_trbl)

    def detect_roi(img_gray):
        frame = detect_frame_corners(img_gray)
        roi = compute_roi_from_frame(frame.corners, rel_margin_trbl, roi_aspect_ratio)

        frame_contour_visualizer = create_frame_contour_visualizer(frame.contour)
        frame_visualizer = create_frame_visualizer(frame.corners)
        roi_visualizer = create_roi_visualizer(
            compute_roi_points(roi.shape, roi.matrix)
        )

        return (
            roi,
            [frame_contour_visualizer, frame_visualizer, roi_visualizer],
        )

    return detect_roi


def create_roi_detector(dimensions_config):
    if "roi" in dimensions_config:
        return create_roi_detector_manual(dimensions_config)
    else:
        return create_roi_detector_from_frame(dimensions_config)


def create_scan_result_transformer_internal(rotate, flip_h, flip_v):
    def scan_result_transformer(scan_result):
        if rotate is not None:
            scan_result = np.rot90(scan_result, (360 - rotate) / 90)
        if flip_h:
            scan_result = np.fliplr(scan_result)
        if flip_v:
            scan_result = np.flipud(scan_result)
        return scan_result.tolist()

    return scan_result_transformer


def create_scan_result_transformer(notify_config):
    rotate = notify_config["rotate"]
    flip_h = notify_config["flipH"]
    flip_v = notify_config["flipV"]
    return create_scan_result_transformer_internal(rotate, flip_h, flip_v)


def select_capture_source(camera_config):
    if "id" in camera_config:
        return camera_config["id"]
    else:
        return camera_config["filename"]


def setup_video_capture(camera_config):
    source = select_capture_source(camera_config)
    capture = cv2.VideoCapture(source)

    if "fourcc" in camera_config:
        s = camera_config["fourcc"]
        fourcc = cv2.VideoWriter_fourcc(s[0], s[1], s[2], s[3])
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)

    if "size" in camera_config:
        [height, width] = camera_config["size"]
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if "fps" in camera_config:
        fps = camera_config["fps"]
        capture.set(cv2.CAP_PROP_FPS, fps)

    if "exposure" in camera_config:
        exposure = camera_config["exposure"]
        capture.set(cv2.CAP_PROP_EXPOSURE, exposure)

    return capture


def create_frame_reader(capture):
    num_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if num_frames == 0:

        def read_frame():
            while True:
                ret, src = capture.read()
                if ret:
                    return src
                else:
                    cv2.waitKey(1)

        return read_frame
    elif num_frames == 1:
        ret, frame = capture.read()
        assert ret, "Reading frame failed."
        return lambda: frame
    else:

        def read_frame():
            while True:
                ret, src = capture.read()
                if not ret:
                    # reach end of stream -> rewind
                    # (can also happen when there is an input error,
                    # but there is no way in OpenCV to tell the difference)
                    # maybe switch to PyAV for capturing
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                    capture.set(cv2.CAP_PROP_POS_MSEC, 0.0)
                    cv2.waitKey(1)
                else:
                    return src

        return read_frame


def remove_gaps(img, grid_shape, rel_gap):
    gap_size = (img.shape[0] * rel_gap[0], img.shape[1] * rel_gap[1])
    padded_img_size = (
        img.shape[0] + gap_size[0],
        img.shape[1] + gap_size[1],
    )
    tile_size_with_gap = (
        padded_img_size[0] / grid_shape[0],
        padded_img_size[1] / grid_shape[1],
    )
    tile_size = (
        tile_size_with_gap[0] - gap_size[0],
        tile_size_with_gap[1] - gap_size[1],
    )
    is_in_tile = (
        np.arange(0, img.shape[0]) % tile_size_with_gap[0] < tile_size[0],
        np.arange(0, img.shape[1]) % tile_size_with_gap[1] < tile_size[1],
    )
    return img[is_in_tile[0], :][:, is_in_tile[1]]


def crop_tile_pixels(img, tag_shape, crop_factors):
    tile_size = (
        img.shape[0] / tag_shape[0],
        img.shape[1] / tag_shape[0],
    )

    is_not_cropped = (
        abs((np.arange(0, img.shape[0]) % tile_size[0]) / tile_size[0] - 0.5)
        <= crop_factors[0],
        abs((np.arange(0, img.shape[1]) % tile_size[1]) / tile_size[1] - 0.5)
        <= crop_factors[0],
    )

    return img[is_not_cropped[0], :][:, is_not_cropped[1]]


class Functor(object):
    def __init__(self, func=lambda: None):
        assert callable(func)
        self.func = func

    def __or__(self, other):
        return Functor(lambda *args, **kwargs: other(self(*args, **kwargs)))

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def split_image(image, grid_shape, tile_shape):
    # splitting image into tiles according to
    # https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    assert len(image.shape) in [2, 3]

    shape = (
        grid_shape[0],
        tile_shape[0],
        grid_shape[1],
        tile_shape[1],
    )
    if len(image.shape) == 3:
        shape = (*shape, image.shape[2])

    return image.reshape(shape).swapaxes(1, 2)


def join_tiles(tiles):
    assert len(tiles.shape) in [4, 5]

    shape = (tiles.shape[0] * tiles.shape[2], tiles.shape[1] * tiles.shape[3])
    if len(tiles.shape) == 5:
        shape = (*shape, tiles.shape[4])

    return tiles.swapaxes(1, 2).reshape(shape)


class Timeout(object):
    def __init__(self, delay):
        self.__delay = delay
        self.__last_reset = 0
        self.reset()

    def reset(self):
        self.__last_reset = time.perf_counter()

    def is_up(self):
        return self.__last_reset + self.__delay < time.perf_counter()
