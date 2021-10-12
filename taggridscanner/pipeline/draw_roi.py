import cv2
import numpy as np

from taggridscanner.utils import Functor


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
        image = cv2.cvtColor(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
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
