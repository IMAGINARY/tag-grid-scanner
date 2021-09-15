import cv2
import math
import numpy as np


def tiles_to_image(tiles, max_value=255, scale_factor=1):
    img = np.zeros(
        (tiles.shape[0] * tiles.shape[2], tiles.shape[1] * tiles.shape[3]),
        dtype=np.uint8,
    )
    for grid_y in range(tiles.shape[0]):
        for grid_x in range(tiles.shape[1]):
            img[
                grid_y * tiles.shape[2] : (grid_y + 1) * tiles.shape[2],
                grid_x * tiles.shape[3] : (grid_x + 1) * tiles.shape[3],
            ] = tiles[grid_y][grid_x]

    img *= max_value

    if scale_factor != 1.0:
        img = cv2.resize(
            img,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )

    return img


class TagDetector:
    def __init__(self, grid_shape, tag_shape, rel_gaps, tags, mirror=False):
        self.grid_shape = grid_shape
        self.rel_gaps = rel_gaps
        self.mirror = mirror
        self.__tag_shape = tag_shape
        self.__tags = tags
        self.__tag_dict, self.__data_for_unknown_tag = self.create_int_tag_dict(tags)

    @property
    def tag_shape(self):
        return self.__tag_shape

    @tag_shape.setter
    def tag_shape(self, s):
        self.__tag_shape = s
        self.__tag_dict, self.__data_for_unknown_tag = self.create_int_tag_dict(
            self.__tags
        )

    @property
    def tags(self):
        return self.__tags

    @tags.setter
    def tags(self, t):
        self.__tags = t
        self.__tag_dict, self.__data_for_unknown_tag = self.create_int_tag_dict(
            self.__tags
        )

    def extract_tiles(self, img):
        tiles = np.zeros((self.grid_shape + self.tag_shape), dtype=np.uint8)
        for grid_y in range(self.grid_shape[0]):
            for grid_x in range(self.grid_shape[1]):
                window = self.tile_window(img, grid_x, grid_y)
                tile = self.reduce_tile(window)
                tiles[grid_y, grid_x] = tile
        return tiles

    def detect_tags(self, tiles):
        detected_tags = np.full((tiles.shape[0], tiles.shape[1]), None, dtype=object)
        for grid_y in range(self.grid_shape[0]):
            for grid_x in range(self.grid_shape[1]):
                tile = tiles[grid_y, grid_x]
                tile_data = self.__tag_dict.get(
                    self.np_tag_to_int(tile), self.__data_for_unknown_tag
                )
                detected_tags[grid_y, grid_x] = tile_data
        return detected_tags

    def create_empty_tags(self):
        return np.full(self.grid_shape, self.__data_for_unknown_tag, dtype=object)

    def string_tag_to_np_tag(self, string_tag):
        return np.fromstring(
            ",".join(list(string_tag)),
            np.uint8,
            self.tag_shape[0] * self.tag_shape[1],
            ",",
        ).reshape(self.tag_shape)

    def np_tag_to_string_tag(self, np_tag):
        return "".join(
            str(e) for e in list(np_tag.reshape(self.tag_shape[0] * self.tag_shape[1]))
        )

    def string_tag_to_int(self, string_tag):
        return int(string_tag, 2)

    def np_tag_to_int(self, np_tag):
        tag_size_linear = self.tag_shape[0] * self.tag_shape[1]
        np_tag_linear = np_tag.reshape(tag_size_linear)
        mask = 1 << tag_size_linear
        int_tag = 0
        for bit in np_tag_linear:
            mask >>= 1
            if bit:
                int_tag |= mask
        return int_tag

    def create_int_tag_dict(self, string_tags):
        data_for_unknown_tag = None
        tag_dict = {}
        for (string_tag, data) in string_tags.items():
            if string_tag == "unknown":
                data_for_unknown_tag = data
            else:
                np_tag = self.string_tag_to_np_tag(string_tag)
                if self.mirror:
                    np_tag = np.fliplr(np_tag)
                tag_dict[self.np_tag_to_int(np_tag)] = data
                np_tag = np.rot90(np_tag)
                tag_dict[self.np_tag_to_int(np_tag)] = data
                np_tag = np.rot90(np_tag)
                tag_dict[self.np_tag_to_int(np_tag)] = data
                np_tag = np.rot90(np_tag)
                tag_dict[self.np_tag_to_int(np_tag)] = data
        return (tag_dict, data_for_unknown_tag)

    def tile_window(self, img, grid_x, grid_y):
        gap_height = img.shape[0] * self.rel_gaps[0]
        img_height_with_added_gap = img.shape[0] + gap_height
        tile_height_with_gap = img_height_with_added_gap / self.grid_shape[0]
        tile_height = tile_height_with_gap - gap_height
        y_start = min(math.floor(grid_y * tile_height_with_gap), img.shape[0] - 1)
        y_end = min(
            math.floor(grid_y * tile_height_with_gap + tile_height), img.shape[0] - 1
        )

        gap_width = img.shape[1] * self.rel_gaps[1]
        img_width_with_added_gap = img.shape[1] + gap_width
        tile_width_with_gap = img_width_with_added_gap / self.grid_shape[1]
        tile_width = tile_width_with_gap - gap_width
        x_start = min(math.floor(grid_x * tile_width_with_gap), img.shape[1] - 1)
        x_end = min(
            math.floor(grid_x * tile_width_with_gap + tile_width), img.shape[1] - 1
        )

        window = img[y_start:y_end, x_start:x_end]

        return window

    def reduce_tile(self, tile_img_gray):
        tile_small = cv2.resize(
            tile_img_gray,
            self.tag_shape,
            interpolation=cv2.INTER_AREA,
        )
        ret, tile_small_bw = cv2.threshold(tile_small, 127, 1, cv2.THRESH_BINARY)
        return tile_small_bw
