import cv2

from taggridscanner.aux.utils import Functor
from taggridscanner.aux.utils import compatible


class Smooth(Functor):
    def __init__(self, smooth):
        super().__init__()
        assert 0.0 <= smooth <= 1.0
        self.smooth = smooth
        self.__last_image = None

    def __call__(self, image):
        if not compatible(self.__last_image, image) or self.smooth == 0.0:
            self.__last_image = image
        else:
            cv2.addWeighted(
                self.__last_image,
                self.smooth,
                image,
                1.0 - self.smooth,
                0.0,
                dst=self.__last_image,
            )
        return self.__last_image
