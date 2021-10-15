import cv2
import numpy as np

from taggridscanner.aux.utils import Functor


class DrawROI(Functor):
    def __init__(self, vertices):
        super().__init__()
        self.vertices = vertices

    @property
    def outline_vertices(self):
        points = np.int32(np.copy(self.vertices))
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
            [self.outline_vertices],
            isClosed=True,
            color=(0, 255, 0),
            thickness=1,
        )
