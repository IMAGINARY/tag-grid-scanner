import sys
import time
import re
import cv2
import numpy as np
from typing import cast
import logging
from copy import deepcopy
from nonblock import nonblock_read

from taggridscanner.aux.config import get_roi_aspect_ratio, set_roi, set_markers, store_config
from taggridscanner.aux.threading import (
    ThreadSafeValue,
    ThreadSafeContainer,
    WorkerThread,
)
from taggridscanner.aux.utils import (
    abs_corners_to_rel_corners,
    rel_corners_to_abs_corners,
    Functor,
    Timeout,
)
from taggridscanner.aux.types import MarkerIdWithCenter, MarkerIdWithCorners, ROIMarkers
from taggridscanner.pipeline.condense_tiles import CondenseTiles
from taggridscanner.pipeline.crop_tile_cells import CropTileCells
from taggridscanner.pipeline.detect_tags import DetectTags
from taggridscanner.pipeline.draw_markers import DrawMarkers
from taggridscanner.pipeline.map_roi import MapROI
from taggridscanner.pipeline.draw_grid import DrawGrid
from taggridscanner.pipeline.draw_roi_editor import DrawROIEditor
from taggridscanner.pipeline.extract_roi import ExtractROI
from taggridscanner.pipeline.retrieve_image import RetrieveImage
from taggridscanner.pipeline.notify import Notify
from taggridscanner.pipeline.preprocess import Preprocess
from taggridscanner.pipeline.track_markers import TrackMarkers
from taggridscanner.pipeline.track_no_markers import TrackNoMarkers
from taggridscanner.pipeline.remove_gaps import RemoveGaps
from taggridscanner.pipeline.threshold import Threshold
from taggridscanner.pipeline.transform_tag_data import TransformTagData
from taggridscanner.pipeline.upscale import Upscale
from taggridscanner.pipeline.view_image import ViewImage

logger = logging.getLogger(__name__)


def clamp_points(points, img_shape):
    for idx in range(0, 4):
        points[idx][0] = max(0, min(points[idx][0], img_shape[1]))
        points[idx][1] = max(0, min(points[idx][1], img_shape[0]))


class ScanWorker(Functor):
    # TODO: Add type hints for other attributes
    src_roi_markers: ROIMarkers
    dst_roi_markers: ROIMarkers

    def __init__(self, config_with_defaults):
        super().__init__(lambda: self.work())
        self.config_with_defaults = config_with_defaults
        self.retrieve_image = RetrieveImage.create_from_config(self.config_with_defaults)
        self.preprocess = Preprocess.create_from_config(self.config_with_defaults)

        self.h, self.w = self.retrieve_image.scaled_size

        if "marker" in self.config_with_defaults["dimensions"]:
            marker_config = self.config_with_defaults["dimensions"]["marker"]
            self.track_markers = TrackMarkers(marker_config["dictionary"], marker_config["tolerance"])
            marker_ids = marker_config["ids"]
            rel_marker_centers = marker_config["centers"]
        else:
            logger.info("No marker config found in the configuration file. Disabling marker tracking.")
            # Use a fake track markers implementation
            self.track_markers = TrackNoMarkers()
            marker_ids = [0, 1, 2, 3]
            rel_marker_centers = [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]

        # Convert the relative marker center coordinates from the marker config to absolute coordinates
        self.src_roi_markers = cast(
            ROIMarkers,
            tuple(
                map(
                    lambda m: MarkerIdWithCorners(m[0], (m[1], m[1], m[1], m[1])),
                    zip(marker_ids, [(c[0] * self.w, c[1] * self.h) for c in rel_marker_centers]),
                )
            ),
        )
        self.dst_roi_markers = self.src_roi_markers

        self.draw_markers = DrawMarkers()

        self.map_roi = MapROI()

        rel_roi_vertices = self.config_with_defaults["dimensions"]["roi"]
        self.idx = 0
        self.vertices = rel_corners_to_abs_corners(rel_roi_vertices, (self.h, self.w))
        self.transformed_vertices = deepcopy(self.vertices)
        logger.debug("Initial vertices: %s", self.transformed_vertices)
        logger.debug("Initial transformed vertices: %s", self.vertices)

        self.draw_roi_editor = DrawROIEditor(active_vertex=self.idx)

        self.extract_roi = ExtractROI(target_aspect_ratio=get_roi_aspect_ratio(self.config_with_defaults))

        grid_shape = self.config_with_defaults["dimensions"]["grid"]
        tag_shape = self.config_with_defaults["dimensions"]["tile"]
        rel_gap = self.config_with_defaults["dimensions"]["gap"]
        crop_factors = self.config_with_defaults["dimensions"]["crop"]
        is_tag_key = lambda entry: re.match("^((unknown)|([01]+))$", entry[0]) is not None
        tags = dict(filter(is_tag_key, self.config_with_defaults["tags"].items()))
        min_contrast = self.config_with_defaults["misc"]["minContrast"]
        detect_rotations = self.config_with_defaults["tags"]["autoRotate"]

        self.remove_gaps = RemoveGaps(grid_shape, tag_shape, rel_gap)
        self.crop_tile_pixels = CropTileCells(grid_shape, tag_shape, crop_factors)
        self.condense_tiles = CondenseTiles(grid_shape, tag_shape)
        self.threshold = Threshold(grid_shape, tag_shape, min_contrast)
        self.detect_tags = DetectTags(grid_shape, tag_shape, tags, detect_rotations=detect_rotations)
        self.upscale = Upscale(10)
        self.draw_grid = DrawGrid(grid_shape, tag_shape, crop_factors)
        self.draw_grid_no_crop = DrawGrid(grid_shape, tag_shape, (1, 1))

        transform_tag_data = TransformTagData.create_from_config(config_with_defaults)
        notify = Notify.create_from_config(config_with_defaults)
        self.transform_and_notify = transform_tag_data | notify

        self.last_tag_data = self.detect_tags.create_empty_tags()

        self.__key = ThreadSafeContainer()
        self.__compute_visualization = ThreadSafeValue(True)

        self.__tag_data = ThreadSafeContainer()
        self.__data_for_config_export = ThreadSafeValue(
            {
                "roi": rel_roi_vertices,
                "markers": self.dst_roi_markers,
            }
        )
        self.__viz = ThreadSafeContainer()
        self.__notify = ThreadSafeValue(True)

        self.preprocessed_src = None
        self.__freeze_input_image = ThreadSafeValue(False)

    @property
    def key(self):
        return self.__key

    @property
    def tag_data(self):
        return self.__tag_data

    @property
    def data_for_config_export(self):
        return self.__data_for_config_export

    @property
    def viz(self):
        return self.__viz

    @property
    def notify(self):
        return self.__notify

    @property
    def compute_visualization(self):
        return self.__compute_visualization

    @property
    def freeze_input_image(self):
        return self.__freeze_input_image

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
        start_ts = time.perf_counter()

        transformed_vertices_need_update = False
        needs_update = False

        try:
            key = self.key.retrieve_nowait()
            vert_step_small = 0.25
            vert_step_big = 10.0

            def offset_current_vertex(offset_2d):
                nonlocal transformed_vertices_need_update
                transformed_vertices_need_update = True
                self.vertices[self.idx][0] += offset_2d[0]
                self.vertices[self.idx][1] += offset_2d[1]

            def next_vertex():
                self.idx = (self.idx + 1) % 4

            def reset_vertices():
                self.vertices = self.default_vertices()

            key_actions = {
                ord("w"): lambda: offset_current_vertex((0, -vert_step_small)),
                ord("a"): lambda: offset_current_vertex((-vert_step_small, 0)),
                ord("s"): lambda: offset_current_vertex((0, +vert_step_small)),
                ord("d"): lambda: offset_current_vertex((+vert_step_small, 0)),
                ord("W"): lambda: offset_current_vertex((0, -vert_step_big)),
                ord("A"): lambda: offset_current_vertex((-vert_step_big, 0)),
                ord("S"): lambda: offset_current_vertex((0, +vert_step_big)),
                ord("D"): lambda: offset_current_vertex((+vert_step_big, 0)),
                32: next_vertex,
                ord("c"): reset_vertices,
            }

            if key in key_actions:
                logger.debug("Key pressed: %s", key)
                key_actions[key]()
                needs_update = True

        except ThreadSafeContainer.Empty:
            pass

        clamp_points(self.vertices, (self.h, self.w))

        rel_corners = abs_corners_to_rel_corners(self.vertices, (self.h, self.w))

        self.draw_roi_editor.active_vertex_idx = self.idx
        self.draw_roi_editor.rel_vertices = rel_corners

        freeze_input_image = self.freeze_input_image.get_nowait()
        compute_visualization = self.compute_visualization.get_nowait()

        if self.preprocessed_src is None or not freeze_input_image:
            src = self.retrieve_image()
            self.preprocessed_src = self.preprocess(src)
            needs_update = True

        copy_preprocessed_src = freeze_input_image or compute_visualization

        if needs_update:
            preprocessed = np.copy(self.preprocessed_src) if copy_preprocessed_src else self.preprocessed_src

            # Track markers providing the markers detected in the last iteration.
            # Searching at the previously detected marker positions first will usually speed up the detection.
            dst_roi_markers, markers_for_viz = self.track_markers(preprocessed, self.dst_roi_markers)

            if transformed_vertices_need_update:
                self.transformed_vertices = self.map_roi(
                    tuple(m.center for m in self.dst_roi_markers),
                    tuple(m.center for m in self.src_roi_markers),
                    self.vertices,
                )
                logger.debug("Source ROI vertices changed. Updating ROI vertices to: %s", self.transformed_vertices)

            if dst_roi_markers is not None:
                self.dst_roi_markers = dst_roi_markers
                self.transformed_vertices = self.map_roi(
                    tuple(m.center for m in self.dst_roi_markers),
                    tuple(m.center for m in self.src_roi_markers),
                    self.vertices,
                )
                logger.debug("All markers found. Updating ROI vertices to: %s", self.transformed_vertices)

            self.__data_for_config_export = ThreadSafeValue(
                {
                    "roi": [[v[0] / self.w, v[1] / self.h] for v in self.transformed_vertices],
                    "markers": [
                        MarkerIdWithCenter(m.id, (m.center[0] / self.w, m.center[1] / self.h))
                        for m in self.dst_roi_markers
                    ],
                }
            )

            extracted_roi = self.extract_roi(preprocessed, self.transformed_vertices)
            gaps_removed = self.remove_gaps(extracted_roi)
            cropped = self.crop_tile_pixels(gaps_removed)
            condensed = self.condense_tiles(cropped)
            thresholded = self.threshold(condensed)
            tag_data = self.detect_tags(thresholded)
            self.tag_data.set(tag_data)

            if not np.array_equal(self.last_tag_data, tag_data):
                self.last_tag_data = tag_data
                if self.notify.get():
                    self.transform_and_notify(tag_data)

            if compute_visualization:
                preprocessed_with_markers = self.draw_markers(
                    preprocessed,
                    markers_for_viz["matched"],
                    markers_for_viz["remaining"],
                    markers_for_viz["not_on_hull"],
                )
                roi_editor_img = self.draw_roi_editor(preprocessed_with_markers, self.transformed_vertices)
                gaps_removed_with_grid = self.draw_grid(gaps_removed)
                cropped_with_grid = self.draw_grid_no_crop(cropped)
                condensed_with_grid = self.draw_grid_no_crop(self.upscale(condensed))
                thresholded_with_grid = self.draw_grid_no_crop(self.upscale(thresholded))
                self.viz.set(
                    (
                        roi_editor_img,
                        extracted_roi,
                        gaps_removed_with_grid,
                        cropped_with_grid,
                        condensed_with_grid,
                        thresholded_with_grid,
                    )
                )

        end_ts = time.perf_counter()
        rate = 1.0 / (end_ts - start_ts)
        # print("max. {:.1f} detections per second".format(rate), file=sys.stderr)


def has_entered_newline():
    """Reads all data currently available on :py:attr:`sys.stdin` until the first newline character. Returns whether a newline character was among the data read."""
    data = "0"  # fake data to enter the loop
    while not data == "" and data is not None:
        if data == "\n":
            return True
        data = nonblock_read(sys.stdin, 1, "t")
    return False


def scan(args):
    config_with_defaults = args["config-with-defaults"]
    if args["ignore_scale"]:
        config_with_defaults["camera"]["scale"] = [1.0, 1.0]

    view_roi_editor = ViewImage("select roi")
    view_extracted_roi = ViewImage("extracted roi")
    view_roi_without_gaps = ViewImage("gaps removed")
    view_cropped_tile_cells = ViewImage("cropped tile pixels")
    view_condensed_cells = ViewImage("condensed")
    view_thresholded = ViewImage("thresholded")
    all_viewers = [
        view_roi_editor,
        view_extracted_roi,
        view_roi_without_gaps,
        view_cropped_tile_cells,
        view_condensed_cells,
        view_thresholded,
    ]

    max_fps = 60
    has_window = False

    if not args["no_gui"]:
        print("Press ENTER to hide/show the UI.", file=sys.stderr)

    auto_hide_timeout = Timeout(args["auto_hide_gui"])

    roi_worker = ScanWorker(config_with_defaults)
    roi_worker.compute_visualization.set(not args["hide_gui"] and not args["no_gui"])
    roi_worker.notify.set(not args["no_notify"])
    producer = WorkerThread(roi_worker)
    producer.rate_limit = args["rate_limit"]
    producer.start()
    roi_worker.tag_data.wait()

    freeze_input_image = roi_worker.freeze_input_image.get_nowait()
    mode = "edit_roi"
    force_ui_update = False
    last_viz = None
    while True:
        frame_start_ts = time.perf_counter()

        try:
            last_viz = viz = roi_worker.viz.retrieve_nowait()
        except ThreadSafeContainer.Empty:
            viz = last_viz if force_ui_update else None
            pass

        if viz is not None:
            (
                roi_editor_img,
                extracted_roi,
                gaps_removed_with_grid,
                cropped_with_grid,
                condensed_with_grid,
                thresholded_with_grid,
            ) = viz

            view_roi_editor(roi_editor_img)
            view_extracted_roi(extracted_roi)
            view_roi_without_gaps(gaps_removed_with_grid)
            view_cropped_tile_cells(cropped_with_grid)
            view_condensed_cells(condensed_with_grid)
            view_thresholded(thresholded_with_grid)
            has_window = True
            force_ui_update = False

        frame_end_ts = time.perf_counter()
        frame_time_left = max(0.0, 1.0 / max_fps - (frame_end_ts - frame_start_ts))

        if not args["no_gui"] and has_entered_newline():
            auto_hide_timeout.reset()
            with roi_worker.compute_visualization.condition:
                show_ui = not roi_worker.compute_visualization.get()
                roi_worker.compute_visualization.set(show_ui)
                force_ui_update = True if show_ui else False

        if auto_hide_timeout.is_up():
            auto_hide_timeout.reset()
            roi_worker.compute_visualization.set(False)

        if not roi_worker.compute_visualization.get_nowait():
            for view_image in all_viewers:
                view_image.hide()
            cv2.pollKey()
            has_window = False

        if has_window:
            ms_to_wait_for_key = max(1, int(1000 * frame_time_left))
            key = cv2.waitKey(ms_to_wait_for_key)
            if key == -1:
                pass
            elif key == ord("f"):
                freeze_input_image = not freeze_input_image
                roi_worker.freeze_input_image.set(freeze_input_image)
                print("Freezing" if freeze_input_image else "Unfreezing", file=sys.stderr)
            else:
                auto_hide_timeout.reset()
                if mode == "edit_roi":
                    if key == 27:  # <ESC>
                        print("Aborting.", file=sys.stderr)
                        sys.exit(1)
                    elif key == 13:  # <ENTER>
                        print(
                            "Press ENTER to save ROI to config file: {}".format(args["config-path"]),
                            file=sys.stderr,
                        )
                        print("Press any other key to abort.", file=sys.stderr)
                        mode = "store_config_data"
                    roi_worker.key.set(key)
                elif mode == "store_config_data":
                    if key == 13:  # <ENTER>
                        print(
                            "Saving ROI and marker centers to: {}".format(args["config-path"]),
                            file=sys.stderr,
                        )
                        data_for_config_export = roi_worker.data_for_config_export.get()
                        logger.info("Updating ROI in config file.")
                        set_roi(args["raw-config"], data_for_config_export["roi"])
                        if "dimensions" in args["raw-config"] and "marker" in args["raw-config"]["dimensions"]:
                            logger.info("Updating markers in config file.")
                            set_markers(args["raw-config"], data_for_config_export["markers"])
                        else:
                            logger.info("Config file has no markers section. Skipping marker update.")
                        store_config(args["raw-config"], args["config-path"])
                    else:
                        print("Aborting.", file=sys.stderr)
                    mode = "edit_roi"
        else:
            time.sleep(frame_time_left)
