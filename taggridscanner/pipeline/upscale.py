import cv2

from taggridscanner.aux.utils import Functor


class Upscale(Functor):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def __call__(self, image):
        return cv2.resize(
            image,
            (image.shape[0] * self.factor, image.shape[1] * self.factor),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
