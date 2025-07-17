from typing import Union

import cv2
from cv2 import aruco
import numpy as np
import math
import logging

from taggridscanner.aux.utils import Functor
from taggridscanner.aux.types import ROIMarkers, MarkerIdWithCorners, MarkersForVis

logger = logging.getLogger(__name__)

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


class TrackMarkers(Functor):
    # TODO: Add "create_from_config" method to create an instance from a configuration file.
    def __init__(self, marker_dict, tolerance: int):
        super().__init__()
        self.detector = aruco.ArucoDetector(marker_dict, ARUCO_PARAMETERS)
        self.tolerance = tolerance

    def __call__(self, image, roi_markers: ROIMarkers) -> (Union[ROIMarkers, None], MarkersForVis):
        """
        Attempts to detect the given markers in the provided image, prioritizing their previous positions.

        If redetection fails, it will attempt to detect all markers in the image, compute their convex hull, try to
        match them with the expected markers.

        Args:
            image: The image in which to search for markers.
            roi_markers (ROIMarkers): The tuple of markers (with IDs and corners) expected to be found.

        Returns:
            Tuple[Union[ROIMarkers, None], MarkersForVis]:
                - If all markers are redetected, returns the tuple of matched markers and a visualization dictionary.
                - If not all markers are found, returns None and a dictionary with matched, remaining, and not_on_hull markers.
        """
        roi_marker_ids = [m.id for m in roi_markers]
        logger.debug("Searching for marker ids: %s", roi_marker_ids)

        # Try to redetect known markers in the image.
        redetected_markers = [self.redetect_marker(image, m, self.tolerance) for m in roi_markers]
        if not (None in redetected_markers):
            logger.debug("All markers were redetected successfully close to their previous positions.")
            return tuple(redetected_markers), {"matched": redetected_markers, "remaining": [], "not_on_hull": []}

        marker_cornerss, marker_ids, _ = self.detector.detectMarkers(image)

        if marker_ids is None or len(marker_cornerss) == 0:
            logger.debug("No markers detected in the image.")
            return None, {"matched": [], "remaining": [], "not_on_hull": []}

        assert len(marker_cornerss) == len(
            marker_ids), "There must be a one-to-one correspondence between list of marker corners and marker ids."

        # Create data structure holding connections between marker ids and corners
        # removing markers that are not in the list of marker ids.
        points = []
        markers_all = []
        logger.debug("Marker corners: %s, Marker ids: %s", marker_cornerss, marker_ids)
        for [marker_corners], [marker_id] in zip(marker_cornerss, marker_ids):
            if marker_id in roi_marker_ids:
                assert len(marker_corners) == 4, "There must be exactly 4 marker corners."
                marker_corners_tuple = tuple(map(lambda p: (p[0], p[1]), marker_corners))
                markers_all.append(MarkerIdWithCorners(marker_id, marker_corners_tuple))
                points.extend(marker_corners)
        logger.debug("All markers: %s", markers_all)

        # Compute convex hull of all marker corners
        logger.debug("All points: %s", points)
        hull = cv2.convexHull(np.array(points))
        logger.debug("Hull: %s", hull)

        # Filter out markers that don't have a corner in the convex hull
        markers_on_hull = []
        markers_not_on_hull = []
        for m in markers_all:
            if next(filter(lambda c: c in hull, m.corners), None) is not None:
                markers_on_hull.append(m)
            else:
                markers_not_on_hull.append(m)
        logger.debug("Markers on hull: %s", markers_on_hull)
        logger.debug("Markers not on hull: %s", markers_not_on_hull)

        matched_markers = list(
            map(lambda m_id: next((m for m in markers_on_hull if m_id == m.id), None), roi_marker_ids))
        logger.debug("Matched markers: %s", matched_markers)

        remaining_markers = list(filter(lambda m: m not in matched_markers, markers_on_hull))
        logger.debug("Remaining markers: %s", remaining_markers)

        if None in matched_markers:
            unmatched_ids = [id_and_match[0] for id_and_match in zip(roi_marker_ids, matched_markers) if
                             id_and_match[1] is None]
            logger.warning("Not all markers were detected (missing ids: %s).", unmatched_ids)
            matched_markers_without_none = list(filter(lambda m: m is not None, matched_markers))
            return None, {"matched": matched_markers_without_none, "remaining": remaining_markers,
                          "not_on_hull": markers_not_on_hull}

        return tuple(matched_markers), {"matched": matched_markers, "remaining": remaining_markers,
                                        "not_on_hull": markers_not_on_hull}

    def redetect_marker(self, image, marker: MarkerIdWithCorners, tolerance: float) \
            -> Union[MarkerIdWithCorners, None]:
        """
        Redetects a known marker in the neighborhood of its original positions.

        :param marker: The marker to be redetected.
        :param tolerance: The tolerance to enlarge the search area around the marker.
            The dimensions of the search area will be (1+tolerance) times the original marker size.
        :return: The redetected marker or None if not found.
        """

        logger.debug("Marker corners as array: %s", np.asarray(marker.corners, dtype=np.float32))

        # Create a slice of the image around the marker's center, enlarged by the tolerance.
        x, y, w, h = cv2.boundingRect(np.asarray(marker.corners, dtype=np.float32))

        # Get the image dimensions.
        iw = image.shape[1]
        ih = image.shape[0]

        # Calculate the new bounding box coordinates with the given tolerance.
        ex = math.floor(x - 0.5 * w * tolerance)
        ey = math.floor(y - 0.5 * h * tolerance)
        ew = math.ceil(w * (1 + tolerance))
        eh = math.ceil(h * (1 + tolerance))

        # Ensure the bounding box is within the image dimensions.
        clamp = lambda v, v_min, v_max: max(v_min, min(v, v_max))
        ecx = clamp(ex, 0, iw - 1)
        ecy = clamp(ey, 0, ih - 1)
        ecw = clamp(ex + ew, 0, iw) - ex
        ech = clamp(ey + eh, 0, ih) - ey

        logger.debug("Attempting redetection of marker %s in slice %s (tolerance: %f, unclipped: %s)",
                     marker.id, (ecx, ecy, ecw, ech), tolerance, (ex, ey, ew, eh))

        # Extract the region of interest from the image.
        roi = image[ecy:ecy + ech, ecx:ecx + ecw]

        # Detect markers in the ROI.
        marker_cornerss, marker_ids, _ = self.detector.detectMarkers(roi)

        # If no markers were detected, return None.
        if marker_ids is None or len(marker_cornerss) == 0:
            logger.debug("No markers detected in the region around marker %s.", marker.id)
            return None

        assert len(marker_cornerss) == len(
            marker_ids), "There must be a one-to-one correspondence between list of marker corners and marker ids."

        # Iterate through detected markers to find the one that matches the original marker.
        # If a match is found, return the marker with its original corners.
        for [marker_corners], [marker_id] in zip(marker_cornerss, marker_ids):
            if marker_id == marker.id:
                assert len(marker_corners) == 4, "There must be exactly 4 marker corners."
                marker_corners_tuple = tuple(map(lambda p: (p[0] + ex, p[1] + ey), marker_corners))
                logger.debug("Redetected marker %s at %s", marker_id, marker_corners_tuple)
                return MarkerIdWithCorners(marker_id, marker_corners_tuple)

        # If no match is found, return None.
        logger.debug("Marker %s not found in the redetection region.", marker.id)
        return None
