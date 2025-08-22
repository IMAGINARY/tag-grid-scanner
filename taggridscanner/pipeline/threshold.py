import cv2

from taggridscanner.aux.utils import Functor, split_image, join_tiles


class Threshold(Functor):
    def __init__(self, grid_shape, tag_shape, min_contrast):
        super().__init__()
        self.grid_shape = grid_shape
        self.tag_shape = tag_shape
        self.min_contrast = min_contrast

    def __call__(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = image.copy()

        tiles = split_image(image, self.grid_shape, self.tag_shape)

        for y in range(0, tiles.shape[0]):
            for x in range(0, tiles.shape[1]):
                tile = tiles[y, x]
                min_val, max_val, _, _ = cv2.minMaxLoc(tile)
                if abs(min_val - max_val) < self.min_contrast * 255:
                    # When the contrast within the tile is low, it probably doesn't resemble any actual tag.
                    # To avoid false positives caused by noise, we set the tile to uniform gray, which
                    # does not match any tag defined via the configuration file after binarization to 0 or 1.
                    # (Tags that are all white or all black are not allowed in the configuration file.)
                    # Using gray also makes affected cells visible in the GUI.
                    tile.fill(127)
                else:
                    cv2.threshold(tile, 0, 255, dst=tile, type=cv2.THRESH_OTSU)

        return join_tiles(tiles)
