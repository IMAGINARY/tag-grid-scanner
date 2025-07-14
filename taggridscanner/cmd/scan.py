import sys
import time
import cv2
import numpy as np

from taggridscanner.aux.config import get_roi_aspect_ratio, set_roi, store_config
from taggridscanner.aux.newline_detector import NewlineDetector
from taggridscanner.aux.threading import (
    ThreadSafeValue,
    ThreadSafeContainer,
    WorkerThreadWithResult,
    WorkerThread,
)
from taggridscanner.aux.utils import (
    abs_corners_to_rel_corners,
    rel_corners_to_abs_corners,
    Functor,
    Timeout,
)
from taggridscanner.pipeline.condense_tiles import CondenseTiles
from taggridscanner.pipeline.crop_tile_cells import CropTileCells
from taggridscanner.pipeline.detect_tags import DetectTags
from taggridscanner.pipeline.draw_markers import DrawMarkers
from taggridscanner.pipeline.draw_grid import DrawGrid
from taggridscanner.pipeline.draw_roi_editor import DrawROIEditor
from taggridscanner.pipeline.extract_roi import ExtractROI
from taggridscanner.pipeline.retrieve_image import RetrieveImage
from taggridscanner.pipeline.notify import Notify
from taggridscanner.pipeline.preprocess import Preprocess
from taggridscanner.pipeline.detect_markers import DetectMarkers
from taggridscanner.pipeline.remove_gaps import RemoveGaps
from taggridscanner.pipeline.threshold import Threshold
from taggridscanner.pipeline.transform_tag_data import TransformTagData
from taggridscanner.pipeline.upscale import Upscale
from taggridscanner.pipeline.view_image import ViewImage


def clamp_points(points, img_shape):
    for idx in range(0, 4):
        points[idx][0] = max(0, min(points[idx][0], img_shape[1]))
        points[idx][1] = max(0, min(points[idx][1], img_shape[0]))


class ScanWorker(Functor):
    def __init__(self, config_with_defaults):
        super().__init__(lambda: self.work())
        self.config_with_defaults = config_with_defaults
        self.retrieve_image = RetrieveImage.create_from_config(
            self.config_with_defaults
        )
        self.preprocess = Preprocess.create_from_config(self.config_with_defaults)

        self.detect_markers = DetectMarkers(
            self.config_with_defaults["dimensions"]["marker"]["dictionary"],
            tuple(self.config_with_defaults["dimensions"]["marker"]["ids"]),
            tuple(map(tuple, self.config_with_defaults["dimensions"]["marker"]["centers"])),
        )

        self.draw_markers = DrawMarkers()

        self.h, self.w = self.retrieve_image.size

        rel_roi_vertices = self.config_with_defaults["dimensions"]["roi"]
        self.idx = 0
        self.vertices = rel_corners_to_abs_corners(rel_roi_vertices, (self.h, self.w))

        self.draw_roi_editor = DrawROIEditor(
            rel_vertices=self.vertices, active_vertex=self.idx
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

        transform_tag_data = TransformTagData.create_from_config(config_with_defaults)
        notify = Notify.create_from_config(config_with_defaults)
        self.transform_and_notify = transform_tag_data | notify

        self.last_tag_data = self.detect_tags.create_empty_tags()

        self.__key = ThreadSafeContainer()
        self.__compute_visualization = ThreadSafeValue(True)

        self.__tag_data = ThreadSafeContainer()
        self.__rel_corners = ThreadSafeValue(rel_roi_vertices)
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
    def rel_corners(self):
        return self.__rel_corners

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

        needs_update = False

        try:
            key = self.key.retrieve_nowait()
            vert_step_small = 0.25
            vert_step_big = 10.0

            def offset_current_vertex(offset_2d):
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
                key_actions[key]()
                needs_update = True

        except ThreadSafeContainer.Empty:
            pass

        clamp_points(self.vertices, (self.h, self.w))

        rel_corners = abs_corners_to_rel_corners(self.vertices, (self.h, self.w))
        self.rel_corners.set(rel_corners)

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
            preprocessed = (
                np.copy(self.preprocessed_src)
                if copy_preprocessed_src
                else self.preprocessed_src
            )

            # TODO: Take care of case where the config file does not contain the marker specification.
            marker_homography_matrix, matched_markers, remaining_markers, markers_not_on_hull = self.detect_markers(
                preprocessed)

            self.extract_roi.rel_corners = rel_corners
            extracted_roi = self.extract_roi(preprocessed, marker_homography_matrix)
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
                # TODO: Take care of case where the config file does not contain the marker specification.
                preprocessed_with_markers = self.draw_markers(
                    preprocessed, matched_markers, remaining_markers, markers_not_on_hull
                )
                roi_editor_img = self.draw_roi_editor(preprocessed_with_markers, marker_homography_matrix)
                gaps_removed_with_grid = self.draw_grid(gaps_removed)
                cropped_with_grid = self.draw_grid_no_crop(cropped)
                condensed_with_grid = self.draw_grid_no_crop(self.upscale(condensed))
                thresholded_with_grid = self.draw_grid_no_crop(
                    self.upscale(thresholded)
                )
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

    newline_detector = NewlineDetector()
    if not args["no_gui"]:
        newline_detector.start()
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

        if not args["no_gui"]:
            try:
                newline_detector.result.retrieve_nowait()
                auto_hide_timeout.reset()
                with roi_worker.compute_visualization.condition:
                    show_ui = not roi_worker.compute_visualization.get()
                    roi_worker.compute_visualization.set(show_ui)
                    force_ui_update = True if show_ui else False
            except ThreadSafeContainer.Empty:
                pass

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
                print(
                    "Freezing" if freeze_input_image else "Unfreezing", file=sys.stderr
                )
            else:
                auto_hide_timeout.reset()
                if mode == "edit_roi":
                    if key == 27:  # <ESC>
                        print("Aborting.", file=sys.stderr)
                        sys.exit(1)
                    elif key == 13:  # <ENTER>
                        print(
                            "Press ENTER to save ROI to config file: {}".format(
                                args["config-path"]
                            ),
                            file=sys.stderr,
                        )
                        print("Press any other key to abort.", file=sys.stderr)
                        mode = "store_roi"
                    roi_worker.key.set(key)
                elif mode == "store_roi":
                    if key == 13:  # <ENTER>
                        print(
                            "Saving ROI to: {}".format(args["config-path"]),
                            file=sys.stderr,
                        )
                        set_roi(args["raw-config"], roi_worker.rel_corners.get())
                        store_config(args["raw-config"], args["config-path"])
                    else:
                        print("Aborting.", file=sys.stderr)
                    mode = "edit_roi"
        else:
            time.sleep(frame_time_left)
