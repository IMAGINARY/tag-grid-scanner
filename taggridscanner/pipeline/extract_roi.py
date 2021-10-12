import cv2
from taggridscanner.roi import (
    compute_roi_shape,
    compute_roi_matrix,
    create_unit_frame_corners,
)

from taggridscanner.utils import Functor, rel_corners_to_abs_corners


class ExtractROI(Functor):
    def __init__(self, target_aspect_ratio, rel_corners=create_unit_frame_corners()):
        super().__init__()
        self.target_aspect_ratio = target_aspect_ratio
        self.rel_corners = rel_corners

    def __call__(self, image):
        # compute target ROI size
        abs_corners = rel_corners_to_abs_corners(self.rel_corners, image.shape)
        target_size = compute_roi_shape(abs_corners, self.target_aspect_ratio)

        # compute homography matrix
        mtx = compute_roi_matrix(image.shape, self.rel_corners, target_size)

        return cv2.warpPerspective(image, mtx, target_size, flags=cv2.INTER_AREA)
