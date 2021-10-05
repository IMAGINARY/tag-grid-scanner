import json
import sys
import cv2
import numpy as np

from .utils import (
    create_preprocessor,
    setup_video_capture,
    abs_corners_to_rel_corners,
)


def to_surrounding_quad(points):
    points = np.int32(points)
    points[0][0] -= 1
    points[0][1] -= 1
    points[1][1] -= 1
    points[3][0] -= 1
    return points


def draw_roi(img, points, active_vertex=0):
    quad_points = to_surrounding_quad(points)
    cv2.polylines(img, [quad_points], isClosed=True, color=(0, 255, 0), thickness=1)
    for p in quad_points:
        cv2.circle(img, p, radius=10, color=(0, 255, 0), thickness=2)
    cv2.circle(
        img, quad_points[active_vertex], radius=10, color=(0, 0, 255), thickness=2
    )


def label(img, text, pos, left, top):
    text_size, baseline = cv2.getTextSize(
        text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=2,
    )

    text_x = pos[0] if left else pos[0] - text_size[0]
    text_y = pos[1] + text_size[1] + baseline if top else pos[1] - baseline

    cv2.putText(
        img,
        text,
        (int(text_x), int(text_y)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=2,
    )


def label_roi(img, points):
    quad_points = to_surrounding_quad(points)
    label(img, " 0: {}".format(points[0]), quad_points[0], True, True)
    label(img, "1: {} ".format(points[1]), quad_points[1], False, True)
    label(img, "2: {} ".format(points[2]), quad_points[2], False, False)
    label(img, " 3: {}".format(points[3]), quad_points[3], True, False)


def clamp_points(points, img_shape):
    for idx in range(0, 4):
        points[idx][0] = max(0, min(points[idx][0], img_shape[1]))
        points[idx][1] = max(0, min(points[idx][1], img_shape[0]))


def roi(args, config, config_with_defaults):
    capture = setup_video_capture(config_with_defaults["camera"])
    preprocess = create_preprocessor(config_with_defaults["camera"])

    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    points = np.array(
        [
            [w / 4.0, h / 4.0],
            [3.0 * w / 4.0, h / 4.0],
            [3.0 * w / 4.0, 3.0 * h / 4.0],
            [w / 4.0, 3.0 * h / 4.0],
        ]
    )

    while True:
        ret, src = capture.read()

        if not ret:
            cv2.waitKey(1)

        src = preprocess(src)
        src = cv2.cvtColor(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        draw_roi(src, points, idx)
        label_roi(src, points)

        cv2.imshow("select roi", src)

        key = cv2.waitKey(1)
        if key == -1:
            continue
        elif key == 119:  # w
            points[idx][1] -= 0.25
        elif key == 97:  # a
            points[idx][0] -= 0.25
        elif key == 115:  # s
            points[idx][1] += 0.25
        elif key == 100:  # d
            points[idx][0] += 0.25
        elif key == 87:  # W
            points[idx][1] -= 10.0
        elif key == 65:  # A
            points[idx][0] -= 10.0
        elif key == 83:  # S
            points[idx][1] += 10.0
        elif key == 68:  # D
            points[idx][0] += 10.0
        elif key == 32:  # <SPACE>
            idx = (idx + 1) % 4
        elif key == 27:  # <ESC>
            print("Aborting.", file=sys.stderr)
            sys.exit(1)
        elif key == 13:  # <ENTER>
            print(json.dumps(abs_corners_to_rel_corners(points, src.shape).tolist()))
            sys.exit(0)

        clamp_points(points, src.shape)
