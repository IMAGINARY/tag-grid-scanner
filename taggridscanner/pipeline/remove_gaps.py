import math
import cv2
import numpy as np

from taggridscanner.aux.utils import Functor


class RemoveGaps(Functor):
    def __init__(self, grid_shape, tag_shape, rel_gap):
        super().__init__()
        self.grid_shape = grid_shape
        self.tag_shape = tag_shape
        self.rel_gap = rel_gap

    def __call__(self, image):
        gap_size = (image.shape[0] * self.rel_gap[0], image.shape[1] * self.rel_gap[1])
        padded_img_size = (
            image.shape[0] + gap_size[0],
            image.shape[1] + gap_size[1],
        )
        tile_size_with_gap = (
            padded_img_size[0] / self.grid_shape[0],
            padded_img_size[1] / self.grid_shape[1],
        )
        tile_size = (
            tile_size_with_gap[0] - gap_size[0],
            tile_size_with_gap[1] - gap_size[1],
        )

        tile_size_int = (
            math.ceil(tile_size[0] / self.tag_shape[0]) * self.tag_shape[0],
            math.ceil(tile_size[1] / self.tag_shape[1]) * self.tag_shape[1],
        )
        result_shape = (
            self.grid_shape[0] * tile_size_int[0],
            self.grid_shape[1] * tile_size_int[1],
        )
        assert len(image.shape) in [2, 3]
        if len(image.shape) == 3:
            result_shape = (*result_shape, image.shape[2])
        result = np.zeros(result_shape, dtype=image.dtype)

        # The following is a workaround to make all tiles equal size
        # and deal with non-integer gap sizes properly.
        # This makes computations later in the pipeline more precise
        # that would otherwise be up to two rows/columns of pixels off.
        sx = tile_size_int[1] / tile_size[1]
        sy = tile_size_int[0] / tile_size[0]
        scale_mtx = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float)
        for y in range(0, self.grid_shape[0]):
            for x in range(0, self.grid_shape[1]):
                tx = x * tile_size_with_gap[1]
                ty = y * tile_size_with_gap[0]
                translate_mtx = np.array(
                    [
                        [1, 0, -tx],
                        [0, 1, -ty],
                        [0, 0, 1],
                    ],
                    dtype=np.float,
                )
                mtx = np.matmul(scale_mtx, translate_mtx)[0:2, 0:3]
                result_tile = result[
                    y * tile_size_int[0] : (y + 1) * tile_size_int[0],
                    x * tile_size_int[1] : (x + 1) * tile_size_int[1],
                ]
                cv2.warpAffine(image, mtx, tile_size_int, dst=result_tile)
        return result
