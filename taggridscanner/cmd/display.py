import math
import cv2
import numpy as np
import random

from taggridscanner.pipeline.detect_tags import string_tag_to_np_tag

GAP_COLOR = 256 / 2
BG_COLOR = 256 / 4


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


def create_linear_transformer(scale: tuple[float, float], rotate, flip_h, flip_v):
    rotate_code = get_rotate_code(rotate)
    flip_code = get_flip_code(flip_h, flip_v)

    def linear_transformer(img):
        if scale != (1.0, 1.0):
            img = cv2.resize(img, None, fx=scale[1], fy=scale[0])
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)
        if flip_code is not None:
            img = cv2.flip(img, flip_code)
        return img

    return linear_transformer


def create_inverse_linear_transformer(scale: tuple[float, float], rotate, flip_h, flip_v):
    i_scale = (1.0 / scale[0], 1.0 / scale[1])
    i_rotate = (360 - rotate) % 360
    i_flip_h = flip_h
    i_flip_v = flip_v
    return create_linear_transformer(i_scale, i_rotate, i_flip_h, i_flip_v)


def rotate_tag(tag, rotation=0):
    while rotation > 0:
        tag = np.rot90(tag)
        rotation -= 1
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


def draw_roi(
    img,
    np_tags,
    grid_shape,
    rel_gap,
):
    img_aspect_ratio = img.shape[1] / img.shape[0]

    roi_width_with_gaps = grid_shape[1] + (grid_shape[1] - 1) * rel_gap[1]
    roi_height_with_gaps = grid_shape[0] + (grid_shape[0] - 1) * rel_gap[0]
    roi_aspect_ratio = roi_width_with_gaps / roi_height_with_gaps

    roi_size = (
        (img.shape[0], img.shape[0] * roi_aspect_ratio)
        if img_aspect_ratio > roi_aspect_ratio
        else (img.shape[1] / roi_aspect_ratio, img.shape[1])
    )

    y_start = math.floor(img.shape[0] / 2 - roi_size[0] / 2)
    y_end = math.ceil(img.shape[0] / 2 + roi_size[0] / 2)

    x_start = math.floor(img.shape[1] / 2 - roi_size[1] / 2)
    x_end = math.ceil(img.shape[1] / 2 + roi_size[1] / 2)

    roi_img = img[y_start:y_end, x_start:x_end]
    roi_img.fill(GAP_COLOR)

    draw_pattern(roi_img, np_tags, grid_shape, rel_gap)


def create_img(
    img_size,
    rel_border_size,
    np_tags,
    grid_shape,
    rel_gap,
    rotate,
    flip_h,
    flip_v,
):
    scale = (1.0, 1.0)
    linear_transform = create_linear_transformer(scale, rotate, flip_h, flip_v)
    inverse_linear_transform = create_inverse_linear_transformer(scale, rotate, flip_h, flip_v)

    img = np.full(img_size, BG_COLOR, np.uint8)
    img = inverse_linear_transform(img)

    y_start = math.floor(img.shape[0] * rel_border_size[0])
    y_end = math.ceil(img.shape[0] * (1.0 - rel_border_size[0]))

    x_start = math.floor(img.shape[1] * rel_border_size[1])
    x_end = math.ceil(img.shape[1] * (1.0 - rel_border_size[1]))

    img_without_border = img[y_start:y_end, x_start:x_end]
    draw_roi(img_without_border, np_tags, grid_shape, rel_gap)

    return linear_transform(img)


def display(args):
    config = args["config"]
    config_with_defaults = args["config-with-defaults"]
    img_size = tuple(config_with_defaults["camera"].get("size", [1920, 1080]))
    rotate = config_with_defaults["camera"]["rotate"]
    flip_h = config_with_defaults["camera"]["flipH"]
    flip_v = config_with_defaults["camera"]["flipV"]

    tag_shape = tuple(config_with_defaults["dimensions"]["tile"])
    grid_shape = tuple(config_with_defaults["dimensions"]["grid"])
    rel_gap = tuple(config_with_defaults["dimensions"]["gap"])

    np_tags = [string_tag_to_np_tag(t, tag_shape) for t in config["tags"] if t != "unknown"]

    rel_border_size = (0.1, 0.1)

    while True:
        img = create_img(
            img_size,
            rel_border_size,
            np_tags,
            grid_shape,
            rel_gap,
            rotate,
            flip_h,
            flip_v,
        )
        cv2.imshow("2D code", img)
        key = cv2.waitKey()
        if key == 27:
            break
