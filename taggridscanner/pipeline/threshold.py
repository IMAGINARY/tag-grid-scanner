import cv2

from taggridscanner.aux.utils import Functor, split_image, join_tiles


class Threshold(Functor):
    def __init__(self, grid_shape, tag_shape):
        super().__init__()
        self.grid_shape = grid_shape
        self.tag_shape = tag_shape

    def __call__(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = image.copy()

        tiles = split_image(image, self.grid_shape, self.tag_shape)

        for y in range(0, tiles.shape[0]):
            for x in range(0, tiles.shape[1]):
                tile = tiles[y, x]
                cv2.threshold(tile, 0, 255, dst=tile, type=cv2.THRESH_OTSU)

        return join_tiles(tiles)
