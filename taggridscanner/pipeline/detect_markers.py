import cv2
from cv2 import aruco
import numpy as np

from taggridscanner.aux.utils import Functor

OPENCV_MARKER_DICTS = {
    "ARUCO_OPENCV_4X4_50": aruco.DICT_4X4_50,
    "ARUCO_OPENCV_4X4_100": aruco.DICT_4X4_100,
    "ARUCO_OPENCV_4X4_250": aruco.DICT_4X4_250,
    "ARUCO_OPENCV_4X4_1000": aruco.DICT_4X4_1000,
    "ARUCO_OPENCV_5X5_50": aruco.DICT_5X5_50,
    "ARUCO_OPENCV_5X5_100": aruco.DICT_5X5_100,
    "ARUCO_OPENCV_5X5_250": aruco.DICT_5X5_250,
    "ARUCO_OPENCV_5X5_1000": aruco.DICT_5X5_1000,
    "ARUCO_OPENCV_6X6_50": aruco.DICT_6X6_50,
    "ARUCO_OPENCV_6X6_100": aruco.DICT_6X6_100,
    "ARUCO_OPENCV_6X6_250": aruco.DICT_6X6_250,
    "ARUCO_OPENCV_6X6_1000": aruco.DICT_6X6_1000,
    "ARUCO_OPENCV_7X7_50": aruco.DICT_7X7_50,
    "ARUCO_OPENCV_7X7_100": aruco.DICT_7X7_100,
    "ARUCO_OPENCV_7X7_250": aruco.DICT_7X7_250,
    "ARUCO_OPENCV_7X7_1000": aruco.DICT_7X7_1000,
    "ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    "APRILTAG_16H5": aruco.DICT_APRILTAG_16h5,
    "APRILTAG_25H9": aruco.DICT_APRILTAG_25h9,
    "APRILTAG_36H10": aruco.DICT_APRILTAG_36h10,
    "APRILTAG_36H11": aruco.DICT_APRILTAG_36h11,
}


def get_marker_dict(marker_dict_name):
    if marker_dict_name in OPENCV_MARKER_DICTS:
        return aruco.getPredefinedDictionary(OPENCV_MARKER_DICTS[marker_dict_name])
    else:
        return None


ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX


class MarkerIdWithCorners:
    """Class for storing the corners of an identified marker."""

    def __init__(self, marker_id: int, corners: list[list[float, float]]):
        self.marker_id = marker_id
        self.corners = corners

    marker_id: int
    corners: list[list[float, float]]

    @property
    def center(self):
        return np.mean(self.corners)

    def __str__(self):
        return f"MarkerIdWithCorners(marker_id={self.marker_id}, corners={self.corners}, center={self.center})"

    def __repr__(self):
        return self.__str__()


class DetectMarkers(Functor):
    def __init__(self, marker_dict, marker_ids):
        super().__init__()
        self.marker_ids = marker_ids
        self.detector = aruco.ArucoDetector(marker_dict, ARUCO_PARAMETERS)

    def __call__(self, image):
        marker_cornerss, marker_ids, _ = self.detector.detectMarkers(image)

        # Create data structure holding connections between marker ids and corners
        # removing markers that are not in the list of marker ids.
        points = []
        markers_all = []
        for [marker_corners], [marker_id] in zip(marker_cornerss, marker_ids):
            if marker_id in self.marker_ids:
                markers_all.append(MarkerIdWithCorners(marker_id, marker_corners))
                points.extend(marker_corners)
        print(markers_all)

        # Compute convex hull of all marker corners
        print("all points: {}".format(points))
        hull = cv2.convexHull(np.array(points))
        print("hull: {}".format(hull))

        # Filter out markers that don't have a corner in the convex hull
        # TODO: Take care of type conversions between float and np.float32.
        markers_on_hull = [
            m
            for m in markers_all
            if any(
                [tuple(c) in map(lambda p: (p[0][0], p[0][1]), hull) for c in m.corners]
            )
        ]
        print("markers on hull: {}".format(markers_on_hull))

        # Replace the marker id with the corresponding marker corners or None if the marker has not been found
        corners_for_id = [None for x in self.marker_ids]
        for m in reversed(markers_on_hull):
            corners_for_id[m.marker_id] = m.corners

        return image, corners_for_id
