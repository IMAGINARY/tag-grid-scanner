import time
import numpy as np


def abs_corners_to_rel_corners(abs_corners, img_shape):
    return np.apply_along_axis(
        lambda p: [p[0] / img_shape[1], p[1] / img_shape[0]], 1, abs_corners
    )


def rel_corners_to_abs_corners(rel_corners, img_shape):
    return np.apply_along_axis(
        lambda p: [p[0] * img_shape[1], p[1] * img_shape[0]], 1, rel_corners
    )


class Functor(object):
    def __init__(self, func=lambda: None):
        assert callable(func)
        self.func = func

    def __or__(self, other):
        return Functor(lambda *args, **kwargs: other(self(*args, **kwargs)))

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def split_image(image, grid_shape, tile_shape):
    # splitting image into tiles according to
    # https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    assert len(image.shape) in [2, 3]

    shape = (
        grid_shape[0],
        tile_shape[0],
        grid_shape[1],
        tile_shape[1],
    )
    if len(image.shape) == 3:
        shape = (*shape, image.shape[2])

    return image.reshape(shape).swapaxes(1, 2)


def join_tiles(tiles):
    assert len(tiles.shape) in [4, 5]

    shape = (tiles.shape[0] * tiles.shape[2], tiles.shape[1] * tiles.shape[3])
    if len(tiles.shape) == 5:
        shape = (*shape, tiles.shape[4])

    return tiles.swapaxes(1, 2).reshape(shape)


class Timeout(object):
    def __init__(self, delay):
        self.__delay = delay
        self.__last_reset = 0
        self.reset()

    def reset(self):
        self.__last_reset = time.perf_counter()

    def is_up(self):
        return self.__last_reset + self.__delay < time.perf_counter()
