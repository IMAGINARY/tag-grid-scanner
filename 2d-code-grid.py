import cv2
import numpy as np
import random

from tags import TAGS

# "shape" is always numpy-style (height, width)

GRID_SHAPE = (16, 16)
TAG_SHAPE = (4, 4)
NUM_TAG_ELEMENTS = TAG_SHAPE[1] * TAG_SHAPE[0]

TILE_ELEMENT_SHAPE = (10, 10)
TILE_SHAPE = (
    TAG_SHAPE[0] * TILE_ELEMENT_SHAPE[0],
    TAG_SHAPE[1] * TILE_ELEMENT_SHAPE[1],
)

GAP_VH = (2, 2)
TILE_SIZE_WITH_GAP = (TILE_SHAPE[0] + GAP_VH[0], TILE_SHAPE[1] + GAP_VH[1])

PATTERN_SHAPE = (
    GRID_SHAPE[0] * TILE_SIZE_WITH_GAP[0] - GAP_VH[0],
    GRID_SHAPE[1] * TILE_SIZE_WITH_GAP[1] - GAP_VH[1],
)

MARGIN = 20  # px
MARGIN_TRBL = (MARGIN, MARGIN, MARGIN, MARGIN)  # px

FRAME_HEIGHT = PATTERN_SHAPE[0] + MARGIN_TRBL[0] + MARGIN_TRBL[2]
FRAME_WIDTH = PATTERN_SHAPE[1] + MARGIN_TRBL[1] + MARGIN_TRBL[3]

BORDER_HEIGHT = 53
IMAGE_HEIGHT = PATTERN_SHAPE[0] + MARGIN_TRBL[0] + MARGIN_TRBL[2] + 2 * BORDER_HEIGHT
IMAGE_WIDTH = int((16 * IMAGE_HEIGHT) / 9)
BORDER_WIDTH = int(
    (IMAGE_WIDTH - (PATTERN_SHAPE[1] + MARGIN_TRBL[1] + MARGIN_TRBL[3])) / 2
)

BORDER_SIZE = (BORDER_HEIGHT, BORDER_WIDTH)

print("GRID_SHAPE:", GRID_SHAPE)
print("TILE_SHAPE:", GRID_SHAPE)
print("FRAME_SHAPE:", (FRAME_HEIGHT, FRAME_WIDTH))
print("MARGIN_TRBL:", MARGIN_TRBL)
print("GAP_VH:", GAP_VH)

GAP_COLOR = 256 / 2
FRAME_COLOR = 3 * 256 / 4
BORDER_COLOR = 0


def create_tile_image(tag, rotation=0):
    assert len(tag) == NUM_TAG_ELEMENTS, "Tag size does not match tile size"
    tile_linear = np.ndarray((NUM_TAG_ELEMENTS,), np.uint8)
    for i in range(len(tag)):
        tile_linear[i] = 0 if tag[i] == "0" else 255
    tile = np.reshape(tile_linear, (TAG_SHAPE[1], TAG_SHAPE[0]))
    while rotation > 0:
        tile = np.rot90(tile)
        rotation = rotation - 1
    return tile


def create_pattern():
    pattern = np.full(PATTERN_SHAPE, GAP_COLOR, np.uint8)
    for x in range(GRID_SHAPE[1]):
        for y in range(GRID_SHAPE[0]):
            tag = random.choice(TAGS)
            rotation = random.randint(0, 3)
            tile = create_tile_image(tag, rotation)
            y_pattern = y * TILE_SIZE_WITH_GAP[0]
            x_pattern = x * TILE_SIZE_WITH_GAP[1]
            tile_resized = cv2.resize(tile, TILE_SHAPE, interpolation=cv2.INTER_NEAREST)
            pattern[
                y_pattern : y_pattern + TILE_SHAPE[0],
                x_pattern : x_pattern + TILE_SHAPE[1],
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
