import math
import cv2
import numpy as np
import random

from .tag_detector import string_tag_to_np_tag
from .utils import (
    compute_rel_gap,
    compute_rel_margin_trbl,
    create_linear_transformer,
    create_inverse_linear_transformer,
)

GAP_COLOR = 256 / 2
FRAME_COLOR = 3 * 256 / 4
BORDER_COLOR = 0


def rotate_tag(tag, rotation=0):
    while rotation > 0:
        tag = np.rot90(tag)
        rotation = rotation - 1
    return tag


def draw_tile(img, tag):
    cv2.resize(tag * 255, img.shape[::-1], dst=img, interpolation=cv2.INTER_NEAREST)


def draw_pattern(img, np_tags, grid_shape, rel_gap):
    gap_size = (img.shape[0] * rel_gap[0], img.shape[1] * rel_gap[1])
    tile_size_with_gap = (
        (img.shape[0] + gap_size[0]) / grid_shape[0],
        (img.shape[1] + gap_size[1]) / grid_shape[1],
    )
    tile_size = (
        tile_size_with_gap[0] - gap_size[0],
        tile_size_with_gap[1] - gap_size[1],
    )
    for x in range(grid_shape[1]):
        for y in range(grid_shape[0]):
            rotation = random.randint(0, 3)
            tag = rotate_tag(random.choice(np_tags), rotation)
            y_tile_begin = math.floor(y * tile_size_with_gap[0])
            y_tile_end = math.ceil(y * tile_size_with_gap[0] + tile_size[0])
            x_tile_begin = math.floor(x * tile_size_with_gap[1])
            x_tile_end = math.ceil(x * tile_size_with_gap[1] + tile_size[1])
            tile_img = img[y_tile_begin:y_tile_end, x_tile_begin:x_tile_end]
            draw_tile(tile_img, tag)


def draw_roi(img, np_tags, grid_shape, rel_margin_trbl, rel_gap):
    y_start = math.floor(img.shape[0] * rel_margin_trbl[0])
    y_end = math.ceil(img.shape[0] * (1.0 - rel_margin_trbl[2]))

    x_start = math.floor(img.shape[1] * rel_margin_trbl[3])
    x_end = math.ceil(img.shape[1] * (1.0 - rel_margin_trbl[1]))

    roi_img = img[y_start:y_end, x_start:x_end]
    roi_img.fill(GAP_COLOR)

    draw_pattern(roi_img, np_tags, grid_shape, rel_gap)


def draw_frame(
    img,
    np_tags,
    grid_shape,
    abs_frame_size,
    abs_margin_trbl,
    abs_gap,
):
    img_aspect_ratio = img.shape[1] / img.shape[0]
    frame_aspect_ratio = abs_frame_size[1] / abs_frame_size[0]

    frame_size = (
        (img.shape[0], img.shape[0] * frame_aspect_ratio)
        if img_aspect_ratio > frame_aspect_ratio
        else (img.shape[1] / frame_aspect_ratio, img.shape[1])
    )

    y_start = math.floor(img.shape[0] / 2 - frame_size[0] / 2)
    y_end = math.ceil(img.shape[0] / 2 + frame_size[0] / 2)

    x_start = math.floor(img.shape[1] / 2 - frame_size[1] / 2)
    x_end = math.ceil(img.shape[1] / 2 + frame_size[1] / 2)

    frame_img = img[y_start:y_end, x_start:x_end]
    frame_img.fill(FRAME_COLOR)

    rel_margin_trbl = compute_rel_margin_trbl(abs_frame_size, abs_margin_trbl)

    rel_gap = compute_rel_gap(abs_frame_size, abs_margin_trbl, abs_gap)

    draw_roi(
        frame_img,
        np_tags,
        grid_shape,
        rel_margin_trbl,
        rel_gap,
    )


def create_img(
    img_size,
    rel_border_size,
    np_tags,
    grid_shape,
    abs_frame_size,
    abs_margin_trbl,
    abs_gap_vh,
    rotate,
    flip_h,
    flip_v,
):
    linear_transform = create_linear_transformer(rotate, flip_h, flip_v)
    inverse_linear_transform = create_inverse_linear_transformer(rotate, flip_h, flip_v)

    img = np.full(img_size, BORDER_COLOR, np.uint8)
    img = inverse_linear_transform(img)

    y_start = math.floor(img.shape[0] * rel_border_size[0])
    y_end = math.ceil(img.shape[0] * (1.0 - rel_border_size[0]))

    x_start = math.floor(img.shape[1] * rel_border_size[1])
    x_end = math.ceil(img.shape[1] * (1.0 - rel_border_size[1]))

    img_without_border = img[y_start:y_end, x_start:x_end]
    draw_frame(
        img_without_border,
        np_tags,
        grid_shape,
        abs_frame_size,
        abs_margin_trbl,
        abs_gap_vh,
    )

    return linear_transform(img)


def display(args, config, config_with_defaults):
    img_size = tuple(config_with_defaults["camera"]["size"])
    rotate = config_with_defaults["camera"]["rotate"]
    flip_h = config_with_defaults["camera"]["flipH"]
    flip_v = config_with_defaults["camera"]["flipV"]

    tag_shape = tuple(config_with_defaults["dimensions"]["tile"])
    grid_shape = tuple(config_with_defaults["dimensions"]["grid"])

    abs_frame_size = tuple(config_with_defaults["dimensions"]["size"])
    abs_margin_trbl = tuple(config_with_defaults["dimensions"]["padding"])
    abs_gap = tuple(config_with_defaults["dimensions"]["gap"])

    np_tags = [
        string_tag_to_np_tag(t, tag_shape) for t in config["tags"] if t != "unknown"
    ]

    rel_border_size = (0.1, 0.1)

    while True:
        img = create_img(
            img_size,
            rel_border_size,
            np_tags,
            grid_shape,
            abs_frame_size,
            abs_margin_trbl,
            abs_gap,
            rotate,
            flip_h,
            flip_v,
        )
        cv2.imshow("2D code", img)
        key = cv2.waitKey()
        if key == 27:
            break
