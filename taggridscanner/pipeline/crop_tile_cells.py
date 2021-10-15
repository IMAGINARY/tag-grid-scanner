import math
import cv2
import numpy as np

from taggridscanner.aux.utils import Functor


class CropTileCells(Functor):
    def __init__(self, grid_shape, tag_shape, crop_factors):
        super().__init__()
        self.grid_shape = grid_shape
        self.tag_shape = tag_shape
        self.crop_factors = crop_factors

    def __call__(self, img):
        cell_size = (
            img.shape[0] / (self.grid_shape[0] * self.tag_shape[0]),
            img.shape[1] / (self.grid_shape[0] * self.tag_shape[1]),
        )

        cropped_cell_size = (
            cell_size[0] * self.crop_factors[0],
            cell_size[1] * self.crop_factors[1],
        )

        cropped_cell_size_int = (
            math.ceil(cropped_cell_size[0]),
            math.ceil(cropped_cell_size[1]),
        )

        result_shape = (
            cropped_cell_size_int[0] * self.grid_shape[0] * self.tag_shape[0],
            cropped_cell_size_int[1] * self.grid_shape[1] * self.tag_shape[1],
        )
        assert len(img.shape) in [2, 3]
        if len(img.shape) == 3:
            result_shape = (*result_shape, img.shape[2])
        result = np.zeros(result_shape, dtype=img.dtype)

        for y in range(0, self.grid_shape[0] * self.tag_shape[0]):
            for x in range(0, self.grid_shape[1] * self.tag_shape[1]):
                result_tile = result[
                    y * cropped_cell_size_int[0] : (y + 1) * cropped_cell_size_int[0],
                    x * cropped_cell_size_int[1] : (x + 1) * cropped_cell_size_int[1],
                ]
                cv2.getRectSubPix(
                    img,
                    cropped_cell_size_int,
                    ((x + 0.5) * cell_size[1], (y + 0.5) * cell_size[0]),
                    patch=result_tile,
                )
        return result
