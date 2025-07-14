import cv2
import numpy as np

from taggridscanner.aux.utils import Functor
from taggridscanner.aux.types import Point2f4, Point2f


class MapROI(Functor):
    def __init__(self):
        super().__init__()

    def __call__(self, src_plane: Point2f4, dst_plane: Point2f4, roi_vertices: Point2f4) -> Point2f4:
        """
        Maps the ROI vertices from the initial plane to the new plane.

        :param src_plane: The initial 3D plane defined by the 2D projection of four 3D points.
        :param dst_plane: The new 3D plane defined by the 2D projection of four 3D points.
        :param roi_vertices: The vertices of the ROI to be mapped.
        :return: Mapped ROI vertices in the new plane.
        """

        src_plane_np = np.asarray(src_plane, dtype=np.float32)
        dst_plane_np = np.asarray(dst_plane, dtype=np.float32)
        roi_vertices_np = np.asarray([[v] for v in roi_vertices], dtype=np.float32)

        # Compute a transformation matrix from the initial plane to the new plane
        transformation_matrix_np = cv2.getPerspectiveTransform(dst_plane_np, src_plane_np)

        # Map the ROI vertices using the transformation matrix
        print(roi_vertices_np)
        mapped_roi_np = cv2.perspectiveTransform(roi_vertices_np, transformation_matrix_np)

        assert mapped_roi_np.shape == (4, 1, 2), "Mapped ROI should have shape (4, 1, 2)"
        mapped_roi = tuple((mapped_roi_np[i][0][0], mapped_roi_np[i][0][1]) for i in range(4))

        return mapped_roi
