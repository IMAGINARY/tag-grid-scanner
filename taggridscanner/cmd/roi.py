import json
import sys
import time
import cv2
import numpy as np

from taggridscanner.config import get_roi_aspect_ratio
from taggridscanner.pipeline.condense_tiles import CondenseTiles
from taggridscanner.pipeline.crop_tile_cells import CropTileCells
from taggridscanner.pipeline.detect_tags import DetectTags
from taggridscanner.pipeline.draw_grid import DrawGrid
from taggridscanner.pipeline.draw_roi_editor import DrawROIEditor
from taggridscanner.pipeline.extract_roi import ExtractROI
from taggridscanner.pipeline.image_source import ImageSource
from taggridscanner.pipeline.preprocess import Preprocess
from taggridscanner.pipeline.remove_gaps import RemoveGaps
from taggridscanner.pipeline.threshold import Threshold
from taggridscanner.pipeline.upscale import Upscale
from taggridscanner.pipeline.view_image import ViewImage

from taggridscanner.threading import ThreadSafeContainer, WorkerThread
from taggridscanner.utils import (
    abs_corners_to_rel_corners,
    rel_corners_to_abs_corners,
    save_roi_corners,
    extract_and_preprocess_roi_config,
    Functor,
)


def clamp_points(points, img_shape):
    for idx in range(0, 4):
        points[idx][0] = max(0, min(points[idx][0], img_shape[1]))
        points[idx][1] = max(0, min(points[idx][1], img_shape[0]))


def done(config, rel_corners):
    print(json.dumps(rel_corners.tolist()))
    dim_config = config["dimensions"]
    if "roi" in dim_config and isinstance(dim_config["roi"], str):
        path = dim_config["roi"]
        print("Saving ROI corners to {}".format(path), file=sys.stderr)
        save_roi_corners(rel_corners, path)
    else:
        print("No path specified. Not saving.", file=sys.stderr)


class ROIWorker(Functor):
    def __init__(self, config_with_defaults):
        super().__init__(lambda: self.work())
        self.config_with_defaults = config_with_defaults
        self.image_source = ImageSource.create_from_config(self.config_with_defaults)
        self.preprocess = Preprocess.create_from_config(self.config_with_defaults)

        self.h, self.w = self.image_source.size

        roi_config = extract_and_preprocess_roi_config(
            self.config_with_defaults["dimensions"]
        )

        self.idx = 0
        self.vertices = (
            rel_corners_to_abs_corners(roi_config, (self.h, self.w))
            if roi_config is not None
            else self.default_vertices()
        )
        self.draw_roi_editor = DrawROIEditor(
            vertices=self.vertices, active_vertex=self.idx
        )

        self.extract_roi = ExtractROI(
            target_aspect_ratio=get_roi_aspect_ratio(self.config_with_defaults),
            rel_corners=abs_corners_to_rel_corners(self.vertices, (self.h, self.w)),
        )

        grid_shape = self.config_with_defaults["dimensions"]["grid"]
        tag_shape = self.config_with_defaults["dimensions"]["tile"]
        rel_gap = self.config_with_defaults["dimensions"]["gap"]
        crop_factors = self.config_with_defaults["dimensions"]["crop"]
        tags = self.config_with_defaults["tags"]

        self.remove_gaps = RemoveGaps(grid_shape, tag_shape, rel_gap)
        self.crop_tile_pixels = CropTileCells(grid_shape, tag_shape, crop_factors)
        self.condense_tiles = CondenseTiles(grid_shape, tag_shape)
        self.threshold = Threshold(grid_shape, tag_shape)
        self.detect_tags = DetectTags(
            grid_shape, tag_shape, tags, detect_rotations=True
        )
        self.upscale = Upscale(10)
        self.draw_grid = DrawGrid(grid_shape, tag_shape, crop_factors)
        self.draw_grid_no_crop = DrawGrid(grid_shape, tag_shape, (1, 1))

        self.last_tag_data = self.detect_tags.create_empty_tags()

        self.__key = ThreadSafeContainer()

    @property
    def key(self):
        return self.__key

    def default_vertices(self):
        return np.array(
            [
                [self.w / 4.0, self.h / 4.0],
                [3.0 * self.w / 4.0, self.h / 4.0],
                [3.0 * self.w / 4.0, 3.0 * self.h / 4.0],
                [self.w / 4.0, 3.0 * self.h / 4.0],
            ]
        )

    def work(self):
        try:
            key = self.key.get_nowait()
            if key == -1:
                pass
            elif key == 119:  # w
                self.vertices[self.idx][1] -= 0.25
            elif key == 97:  # a
                self.vertices[self.idx][0] -= 0.25
            elif key == 115:  # s
                self.vertices[self.idx][1] += 0.25
            elif key == 100:  # d
                self.vertices[self.idx][0] += 0.25
            elif key == 87:  # W
                self.vertices[self.idx][1] -= 10.0
            elif key == 65:  # A
                self.vertices[self.idx][0] -= 10.0
            elif key == 83:  # S
                self.vertices[self.idx][1] += 10.0
            elif key == 68:  # D
                self.vertices[self.idx][0] += 10.0
            elif key == 32:  # <SPACE>
                self.idx = (self.idx + 1) % 4
            elif key == 99:  # c
                self.vertices = self.default_vertices()
        except ThreadSafeContainer.Empty:
            pass

        clamp_points(self.vertices, (self.h, self.w))

        src = self.preprocess(self.image_source.read())

        self.extract_roi.rel_corners = abs_corners_to_rel_corners(
            self.vertices, (self.h, self.w)
        )
        extracted_roi = self.extract_roi(src)

        self.draw_roi_editor.vertices = self.vertices
        roi_editor_img = self.draw_roi_editor(src)

        gaps_removed = self.remove_gaps(extracted_roi)

        gaps_removed_with_grid = self.draw_grid(gaps_removed)

        cropped = self.crop_tile_pixels(gaps_removed)
        cropped_with_grid = self.draw_grid_no_crop(cropped)

        condensed = self.condense_tiles(cropped)
        condensed_with_grid = self.draw_grid_no_crop(self.upscale(condensed))

        thresholded = self.threshold(condensed)
        thresholded_with_grid = self.draw_grid_no_crop(self.upscale(thresholded))

        tag_data = self.detect_tags(thresholded)

        if not np.array_equal(self.last_tag_data, tag_data):
            self.last_tag_data = tag_data
            print(tag_data)

        rel_corners = abs_corners_to_rel_corners(self.vertices, (self.h, self.w))

        return (
            roi_editor_img,
            extracted_roi,
            gaps_removed_with_grid,
            cropped_with_grid,
            condensed_with_grid,
            thresholded_with_grid,
            tag_data,
            rel_corners,
        )


def roi(args):
    config_with_defaults = args["config-with-defaults"]

    view_roi_editor = ViewImage("select roi")
    view_extracted_roi = ViewImage("extracted roi")
    view_roi_without_gaps = ViewImage("gaps removed")
    view_cropped_tile_cells = ViewImage("cropped tile pixels")
    view_condensed_cells = ViewImage("condensed")
    view_thresholded = ViewImage("thresholded")

    rel_corners = None
    max_fps = 60
    has_window = False

    roi_worker = ROIWorker(config_with_defaults)
    producer = WorkerThread(roi_worker)
    producer.rate_limit = 4
    producer.start()
    producer.result.wait()
    while True:
        frame_start_ts = time.perf_counter()

        try:
            (
                roi_editor_img,
                extracted_roi,
                gaps_removed_with_grid,
                cropped_with_grid,
                condensed_with_grid,
                thresholded_with_grid,
                tag_data,
                rel_corners,
            ) = producer.result.get_nowait()

            view_roi_editor(roi_editor_img)
            view_extracted_roi(extracted_roi)
            view_roi_without_gaps(gaps_removed_with_grid)
            view_cropped_tile_cells(cropped_with_grid)
            view_condensed_cells(condensed_with_grid)
            view_thresholded(thresholded_with_grid)

            has_window = True
        except ThreadSafeContainer.Empty:
            pass

        frame_end_ts = time.perf_counter()
        frame_time_left = max(0.0, 1.0 / max_fps - (frame_end_ts - frame_start_ts))

        if has_window:
            ms_to_wait_for_key = max(1, int(1000 * frame_time_left))
            key = cv2.waitKey(ms_to_wait_for_key)
            if key == -1:
                pass
            else:
                if key == 27:  # <ESC>
                    print("Aborting.", file=sys.stderr)
                    sys.exit(1)
                elif key == 13:  # <ENTER>
                    if rel_corners is not None:
                        done(
                            config_with_defaults,
                            rel_corners,
                        )
                        sys.exit(0)
                roi_worker.key.set(key)
        else:
            time.sleep(frame_time_left)
