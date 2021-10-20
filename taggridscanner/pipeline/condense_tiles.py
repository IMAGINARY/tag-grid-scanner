import cv2

from taggridscanner.aux.utils import Functor


class CondenseTiles(Functor):
    def __init__(self, grid_shape, tag_shape):
        super().__init__()
        self.grid_shape = grid_shape
        self.tag_shape = tag_shape

    def __call__(self, image):
        target_size = (
            self.grid_shape[0] * self.tag_shape[0],
            self.grid_shape[1] * self.tag_shape[1],
        )
        return cv2.resize(
            image,
            target_size,
            interpolation=cv2.INTER_AREA,
        )
