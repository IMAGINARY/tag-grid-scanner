import cv2
import numpy as np

from taggridscanner.aux.utils import Functor


class DrawROI(Functor):
    def __init__(self, rel_vertices):
        super().__init__()
        self.rel_vertices = rel_vertices

    def abs_vertices(self, shape):
        def rel_to_abs(p):
            return [p[0] * shape[1], p[1] * shape[0]]

        return list(map(rel_to_abs, np.copy(self.rel_vertices)))

    def outline_vertices(self, shape):
        points = np.int32(self.abs_vertices(shape))
        points[0][0] -= 1
        points[0][1] -= 1
        points[1][1] -= 1
        points[3][0] -= 1
        return points

    def __call__(self, image):
        assert len(image.shape) in [2, 3]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.draw_quad(image)
        return image

    def draw_quad(self, image):
        cv2.polylines(
            image,
            [self.outline_vertices(image.shape)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=1,
        )
