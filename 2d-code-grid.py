import cv2
import numpy as np
import random

from tags import TAGS

GRID_SHAPE = (16, 16)  # (height, width)
TILE_SHAPE = (4, 4)  # (height, width)
NUM_TILE_ELEMENTS = TILE_SHAPE[1] * TILE_SHAPE[0]

TILE_ELEMENT_SIZE = (10, 10)
TILE_SIZE = (TILE_SHAPE[0] * TILE_ELEMENT_SIZE[0], TILE_SHAPE[1] * TILE_ELEMENT_SIZE[1])

GAP_SIZE = (4, 4)
TILE_SIZE_WITH_GAP = (TILE_SIZE[0] + GAP_SIZE[0], TILE_SIZE[1] + GAP_SIZE[1])

PATTERN_SIZE = (
    GRID_SHAPE[0] * TILE_SIZE_WITH_GAP[0] - GAP_SIZE[0],
    GRID_SHAPE[1] * TILE_SIZE_WITH_GAP[1] - GAP_SIZE[1],
)

MARGIN = 20  # px
MARGIN_TRBL = (MARGIN, MARGIN, MARGIN, MARGIN)  # px

FRAME_HEIGHT = PATTERN_SIZE[0] + MARGIN_TRBL[0] + MARGIN_TRBL[2]
FRAME_WIDTH = PATTERN_SIZE[1] + MARGIN_TRBL[1] + MARGIN_TRBL[3]

BORDER_HEIGHT = 53
IMAGE_HEIGHT = PATTERN_SIZE[0] + MARGIN_TRBL[0] + MARGIN_TRBL[2] + 2 * BORDER_HEIGHT
IMAGE_WIDTH = int((16 * IMAGE_HEIGHT) / 9)
BORDER_WIDTH = int(
    (IMAGE_WIDTH - (PATTERN_SIZE[1] + MARGIN_TRBL[1] + MARGIN_TRBL[3])) / 2
)

BORDER_SIZE = (BORDER_HEIGHT, BORDER_WIDTH)

print("GRID:", GRID_SHAPE)
print("TILE:", GRID_SHAPE)
print("FRAME:", (FRAME_HEIGHT, FRAME_WIDTH))
print("MARGIN:", MARGIN_TRBL)
print("GAP:", GAP_SIZE)

GAP_COLOR = 256 / 2
FRAME_COLOR = 3 * 256 / 4
BORDER_COLOR = 0


def create_tile_image(tag, rotation=0):
    assert len(tag) == NUM_TILE_ELEMENTS, "Tag size does not match tile size"
    tile_linear = np.ndarray((NUM_TILE_ELEMENTS,), np.uint8)
    for i in range(len(tag)):
        tile_linear[i] = 0 if tag[i] == "0" else 255
    tile = np.reshape(tile_linear, (TILE_SHAPE[1], TILE_SHAPE[0]))
    while rotation > 0:
        tile = np.rot90(tile)
        rotation = rotation - 1
    return tile


def create_pattern():
    pattern = np.full(PATTERN_SIZE, GAP_COLOR, np.uint8)
    for x in range(GRID_SHAPE[1]):
        for y in range(GRID_SHAPE[0]):
            tag = random.choice(TAGS)
            rotation = random.randint(0, 3)
            tile = create_tile_image(tag, rotation)
            y_pattern = y * TILE_SIZE_WITH_GAP[0]
            x_pattern = x * TILE_SIZE_WITH_GAP[1]
            tile_resized = cv2.resize(tile, TILE_SIZE, interpolation=cv2.INTER_NEAREST)
            pattern[
                y_pattern : y_pattern + TILE_SIZE[0],
                x_pattern : x_pattern + TILE_SIZE[1],
            ] = tile_resized
    return pattern


def create_img():
    pattern = create_pattern()
    frame = cv2.copyMakeBorder(
        pattern,
        MARGIN_TRBL[0],
        MARGIN_TRBL[2],
        MARGIN_TRBL[3],
        MARGIN_TRBL[1],
        cv2.BORDER_CONSTANT,
        value=FRAME_COLOR,
    )
    border = cv2.copyMakeBorder(
        frame,
        BORDER_SIZE[0],
        BORDER_SIZE[0],
        BORDER_SIZE[1],
        BORDER_SIZE[1],
        cv2.BORDER_CONSTANT,
        value=BORDER_COLOR,
    )
    return border


# img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), np.uint8)
while True:
    img = create_img()
    cv2.imshow("2D code", img)
    key = cv2.waitKey()
    if key == 27:
        break
