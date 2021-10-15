import cv2
import numpy as np

from taggridscanner.aux.utils import Functor


class GenerateCalibrationPattern(Functor):
    def __init__(self, img_shape, pattern_shape):
        super().__init__()
        self.img_shape = img_shape
        self.pattern_shape = pattern_shape

    def __call__(self):
        img = np.full(self.img_shape, 255, dtype=np.uint8)

        parity = 0
        edge_length = min(
            img.shape[0] / (self.pattern_shape[0] + 1 + 2),
            img.shape[1] / (self.pattern_shape[1] + 1 + 2),
        )
        for y in range(1, self.pattern_shape[0] + 1 + 1):
            y_begin = int(edge_length * y)
            y_end = int(edge_length * (y + 1))
            for x in range(1 + parity, self.pattern_shape[1] + 1 + 1, 2):
                x_begin = int(edge_length * x)
                x_end = int(edge_length * (x + 1))
                cv2.rectangle(
                    img,
                    (x_begin, y_begin),
                    (x_end, y_end),
                    color=0,
                    thickness=cv2.FILLED,
                )
            parity = (parity + 1) % 2
        return img
