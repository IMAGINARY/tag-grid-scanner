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

MarkerIds = (int, int, int, int)
Point2f = (float, float)
Point2f4 = (Point2f, Point2f, Point2f, Point2f)


class MarkerIdWithCorners:
    """Class for storing the corners of an identified marker."""

    def __init__(self, id: int, corners: Point2f4):
        self.id = id
        self.corners = corners
        self.center = (
            (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4,
            (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4
        )

    id: int
    corners: Point2f4
    center: Point2f

    def __str__(self):
        return f"MarkerIdWithCorners(id={self.id}, corners={self.corners}, center={self.center})"

    def __repr__(self):
        return self.__str__()


class DetectMarkers(Functor):
    # TODO: Add "create_from_config" method to create an instance from a configuration file.
    def __init__(self, marker_dict, marker_ids: MarkerIds,
                 rel_marker_centers: Point2f4):
        super().__init__()
        self.marker_ids = marker_ids
        self.rel_marker_centers = rel_marker_centers
        self.detector = aruco.ArucoDetector(marker_dict, ARUCO_PARAMETERS)
        self.prev_homography_matrix = np.identity(3)

    def __call__(self, image) -> (np.ndarray, list[MarkerIdWithCorners], list[MarkerIdWithCorners],
                                  list[MarkerIdWithCorners]):
        print("Searching for marker ids: {}".format(self.marker_ids));

        marker_cornerss, marker_ids, _ = self.detector.detectMarkers(image)

        if marker_ids is None or len(marker_cornerss) == 0:
            return self.prev_homography_matrix, [], [], []

        assert len(marker_cornerss) == len(
            marker_ids), "There must be a one-to-one correspondence between list of marker corners and marker ids."

        # Create data structure holding connections between marker ids and corners
        # removing markers that are not in the list of marker ids.
        points = []
        markers_all = []
        print(marker_cornerss, marker_ids)
        for [marker_corners], [marker_id] in zip(marker_cornerss, marker_ids):
            if marker_id in self.marker_ids:
                assert len(marker_corners) == 4, "There must be exactly 4 marker corners."
                marker_corners_tuple = tuple(map(lambda p: (p[0], p[1]), marker_corners))
                markers_all.append(MarkerIdWithCorners(marker_id, marker_corners_tuple))
                points.extend(marker_corners)
        print(markers_all)

        # Compute convex hull of all marker corners
        print("all points: {}".format(points))
        hull = cv2.convexHull(np.array(points))
        print("hull: {}".format(hull))

        # Filter out markers that don't have a corner in the convex hull
        # TODO: Take care of type conversions between float and np.float32.
        markers_on_hull = []
        markers_not_on_hull = []
        for m in markers_all:
            if next(filter(lambda c: c in hull, m.corners), None) is not None:
                markers_on_hull.append(m)
            else:
                markers_not_on_hull.append(m)
        print("markers on hull: {}".format(markers_on_hull))
        print("markers not on hull: {}".format(markers_not_on_hull))

        matched_markers = list(
            map(lambda m_id: next((m for m in markers_on_hull if m_id == m.id), None), self.marker_ids))
        print("matched markers: {}".format(matched_markers))

        remaining_markers = list(filter(lambda m: m not in matched_markers, markers_on_hull))
        print("remaining markers: {}".format(remaining_markers))

        if None in matched_markers:
            print("Not all markers were detected. Skipping homography computation.")
            matched_markers_without_none = list(filter(lambda m: m is not None, matched_markers))
            return self.prev_homography_matrix, matched_markers_without_none, remaining_markers, markers_not_on_hull

        height, width, _ = image.shape
        abs_marker_centers = list(map(lambda m: (m[0] * width, m[1] * height), self.rel_marker_centers))
        abs_matched_marker_centers = list(map(lambda m: m.center, matched_markers))
        rel_matched_marker_centers = list(map(lambda m: (m.center[0] / width, m.center[1] / height), matched_markers))
        print("original relative  markers centers: {}".format(self.rel_marker_centers))
        print("matched relative markers centers:  {}".format(rel_matched_marker_centers))
        print("original absolute  markers centers: {}".format(abs_marker_centers))
        print("matched absolute markers centers:  {}".format(abs_matched_marker_centers))

        h, status = cv2.findHomography(np.asarray(abs_matched_marker_centers), np.asarray(abs_marker_centers))
        print("homography matrix: {}".format(h))
        print("point status: {}".format(status))
        if h is None or 0 in status.flatten().tolist():
            print(
                "Homography matrix could not be computed or some points were not matched. Skipping homography computation.")
            return image, self.prev_homography_matrix, matched_markers, remaining_markers, markers_not_on_hull

        self.prev_homography_matrix = h
        return h, matched_markers, remaining_markers, markers_not_on_hull
