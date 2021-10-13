import math
import cv2
import numpy as np

from taggridscanner.utils import Functor


class DrawGrid(Functor):
    def __init__(self, grid_shape, tag_shape, crop_factors):
        super().__init__()
        self.grid_shape = grid_shape
        self.tag_shape = tag_shape
        self.crop_factors = crop_factors

    def __call__(self, img):
        assert len(img.shape) in [2, 3]

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        result_shape = (
            img.shape[0] + (self.grid_shape[0] * self.tag_shape[0] - 1),
            img.shape[1] + (self.grid_shape[1] * self.tag_shape[1] - 1),
            img.shape[2],
        )

        result = np.zeros(result_shape, dtype=img.dtype)
        cell_size = (
            img.shape[0] / (self.grid_shape[0] * self.tag_shape[0]),
            img.shape[1] / (self.grid_shape[1] * self.tag_shape[1]),
        )

        crop_offset = (
            math.ceil(cell_size[0] * 0.5 * (1 - self.crop_factors[0])),
            math.ceil(cell_size[1] * 0.5 * (1 - self.crop_factors[1])),
        )
        # cell level
        for y in range(0, self.grid_shape[0] * self.tag_shape[0]):
            y_start = int(cell_size[0] * y)
            y_end = int(cell_size[0] * (y + 1))
            for x in range(0, self.grid_shape[1] * self.tag_shape[1]):
                x_start = int(cell_size[1] * x)
                x_end = int(cell_size[1] * (x + 1))
                result[y_start + y : y_end + y, x_start + x : x_end + x] = img[
                    y_start:y_end, x_start:x_end
                ]

                cv2.rectangle(
                    result,
                    (
                        x_start + x - 1 + crop_offset[1],
                        y_start + y - 1 + crop_offset[0],
                    ),
                    (x_end + x - crop_offset[1], y_end + y - crop_offset[0]),
                    (128, 255, 128),
                )

        # tag level
        for y in range(0, self.grid_shape[0] * self.tag_shape[0]):
            cv2.line(
                result,
                (0, int(y * cell_size[0] + y - 1)),
                (result.shape[1], int(y * cell_size[0] + y - 1)),
                (255, 128, 128),
                thickness=1,
            )
        for x in range(0, self.grid_shape[1] * self.tag_shape[1]):
            cv2.line(
                result,
                (int(x * cell_size[1] + x - 1), 0),
                (int(x * cell_size[1] + x - 1), result.shape[0]),
                (255, 128, 128),
                thickness=1,
            )

        # grid level
        for y in range(0, self.grid_shape[0] * self.tag_shape[0], self.tag_shape[0]):
            cv2.line(
                result,
                (0, int(y * cell_size[0] + y - 1)),
                (result.shape[1], int(y * cell_size[0] + y - 1)),
                (0, 0, 255),
                thickness=1,
            )

        for x in range(0, self.grid_shape[1] * self.tag_shape[1], self.tag_shape[1]):
            cv2.line(
                result,
                (int(x * cell_size[1] + x - 1), 0),
                (int(x * cell_size[1] + x - 1), result.shape[0]),
                (0, 0, 255),
                thickness=1,
            )

        return result
