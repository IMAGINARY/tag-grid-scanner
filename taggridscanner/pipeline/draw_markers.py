import cv2
from cv2 import aruco
import numpy as np

from taggridscanner.aux.utils import Functor

MATCHED_MARKERS_COLOR = (0, 255, 0)
REMAINING_MARKERS_COLOR = (0, 255, 255)
MARKERS_NOT_ON_HULL_COLOR = (0, 0, 255)


class DrawMarkers(Functor):
    def __init__(self):
        super().__init__()

    def draw_markers(self, image, markers, color):
        if len(markers) == 0:
            return

        corners = tuple(np.asarray([marker.corners], dtype=np.float32) for marker in markers)
        ids = np.asarray([marker.id for marker in markers], dtype=np.int32)

        aruco.drawDetectedMarkers(image, corners, ids, color)

    def __call__(self, image, matched_markers, remaining_markers, markers_not_on_hull):
        assert len(image.shape) in [2, 3]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        self.draw_markers(image, remaining_markers, REMAINING_MARKERS_COLOR)
        self.draw_markers(image, markers_not_on_hull, MARKERS_NOT_ON_HULL_COLOR)
        self.draw_markers(image, matched_markers, MATCHED_MARKERS_COLOR)

        return image
