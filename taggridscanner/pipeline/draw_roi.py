import cv2
import numpy as np

from taggridscanner.aux.utils import Functor


class DrawROI(Functor):
    def __init__(self):
        super().__init__()

    def __call__(self, image, roi_vertices):
        assert len(image.shape) in [2, 3]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.draw_quad(image, roi_vertices)
        return image

    @staticmethod
    def outline_vertices(vertices):
        ivertices = np.int32(vertices)
        ivertices[0][0] -= 1
        ivertices[0][1] -= 1
        ivertices[1][1] -= 1
        ivertices[3][0] -= 1
        return ivertices

    def draw_quad(self, image, roi_vertices):
        cv2.polylines(
            image,
            [DrawROI.outline_vertices(roi_vertices)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=1,
        )
