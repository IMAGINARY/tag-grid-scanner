import json
import sys
import cv2
import numpy as np
from taggridscanner.config import get_roi_aspect_ratio
from taggridscanner.pipeline.draw_roi_editor import DrawROIEditor
from taggridscanner.pipeline.extract_roi import ExtractROI
from taggridscanner.pipeline.image_source import ImageSource
from taggridscanner.pipeline.preprocess import Preprocess

from taggridscanner.utils import (
    abs_corners_to_rel_corners,
    rel_corners_to_abs_corners,
    save_roi_corners,
    extract_and_preprocess_roi_config,
)


def clamp_points(points, img_shape):
    for idx in range(0, 4):
        points[idx][0] = max(0, min(points[idx][0], img_shape[1]))
        points[idx][1] = max(0, min(points[idx][1], img_shape[0]))


def done(config, rel_corners):
    print(json.dumps(rel_corners.tolist()))
    dim_config = config["dimensions"]
    if "roi" in dim_config and type(dim_config["roi"]) == str:
        path = dim_config["roi"]
        print("Saving ROI corners to {}".format(path), file=sys.stderr)
        save_roi_corners(rel_corners, path)
    else:
        print("No path specified. Not saving.", file=sys.stderr)


def roi(args):
    config_with_defaults = args["config-with-defaults"]
    image_source = ImageSource.create(config_with_defaults)
    preprocess = Preprocess(config_with_defaults)

    h, w = image_source.size

    def default_vertices():
        return np.array(
            [
                [w / 4.0, h / 4.0],
                [3.0 * w / 4.0, h / 4.0],
                [3.0 * w / 4.0, 3.0 * h / 4.0],
                [w / 4.0, 3.0 * h / 4.0],
            ]
        )

    roi_config = extract_and_preprocess_roi_config(config_with_defaults["dimensions"])

    idx = 0
    vertices = (
        rel_corners_to_abs_corners(roi_config, (h, w))
        if roi_config is not None
        else default_vertices()
    )
    draw_roi_editor = DrawROIEditor(vertices=vertices, active_vertex=idx)

    extract_roi = ExtractROI(
        target_aspect_ratio=get_roi_aspect_ratio(config_with_defaults),
        rel_corners=abs_corners_to_rel_corners(vertices, (h, w)),
    )

    while True:
        src = preprocess(image_source.read())

        extract_roi.rel_corners = abs_corners_to_rel_corners(vertices, (h, w))
        cv2.imshow("extracted roi", extract_roi(src))

        draw_roi_editor.vertices = vertices
        cv2.imshow("select roi", draw_roi_editor(src))

        key = cv2.waitKey(1)
        if key == -1:
            continue
        elif key == 119:  # w
            vertices[idx][1] -= 0.25
        elif key == 97:  # a
            vertices[idx][0] -= 0.25
        elif key == 115:  # s
            vertices[idx][1] += 0.25
        elif key == 100:  # d
            vertices[idx][0] += 0.25
        elif key == 87:  # W
            vertices[idx][1] -= 10.0
        elif key == 65:  # A
            vertices[idx][0] -= 10.0
        elif key == 83:  # S
            vertices[idx][1] += 10.0
        elif key == 68:  # D
            vertices[idx][0] += 10.0
        elif key == 32:  # <SPACE>
            idx = (idx + 1) % 4
        elif key == 99:  # c
            vertices = ()
        elif key == 27:  # <ESC>
            print("Aborting.", file=sys.stderr)
            sys.exit(1)
        elif key == 13:  # <ENTER>
            done(config_with_defaults, abs_corners_to_rel_corners(vertices, src.shape))
            sys.exit(0)

        clamp_points(vertices, src.shape)
