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

        self.last_image_shape = None
        self.operator = None

    def sample_tiles_along_axis(self, image_size_along_axis, axis):
        assert axis in (0, 1), "Axis must be 0 (y) or 1 (x)"

        num_cells = self.grid_shape[axis] * self.tag_shape[axis]
        crop_factor = self.crop_factors[axis]

        cell_size = image_size_along_axis / num_cells
        cropped_cell_size = cell_size * crop_factor
        cropped_cell_size_int = math.ceil(cropped_cell_size)
        half_crop_size = (cell_size - cropped_cell_size) / 2

        samples = np.zeros(num_cells * cropped_cell_size_int, dtype=np.float32)
        for i in range(num_cells):
            start = i * cell_size + half_crop_size
            end = start + cropped_cell_size
            start_idx = i * cropped_cell_size_int
            end_idx = (i + 1) * cropped_cell_size_int
            samples[start_idx:end_idx] = np.linspace(start, end, cropped_cell_size_int, dtype=np.float32)

        return samples

    def create_operator(self, image_shape):
        samples_x = self.sample_tiles_along_axis(image_shape[1], 1)
        samples_y = self.sample_tiles_along_axis(image_shape[0], 0)

        map_x = np.repeat(samples_x.reshape((1, samples_x.shape[0])), samples_y.shape[0], axis=0)
        map_y = np.repeat(samples_y.reshape((samples_y.shape[0], 1)), samples_x.shape[0], axis=1)

        return lambda image: cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    def __call__(self, image):
        if self.last_image_shape != image.shape or self.operator is None:
            self.operator = self.create_operator(image.shape)
            self.last_image_shape = image.shape

        return self.operator(image)
