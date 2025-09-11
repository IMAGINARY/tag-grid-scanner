import cv2

from taggridscanner.aux.utils import Functor


class Grayscale(Functor):
    def __call__(self, image):
        assert len(image.shape) in [2, 3]
        return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
