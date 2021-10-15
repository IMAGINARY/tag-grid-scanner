import numpy as np

from taggridscanner.aux.utils import Functor


class TransformTagData(Functor):
    def __init__(self, rotate, flip_h, flip_v):
        super().__init__()
        self.rotate = rotate
        self.flip_h = flip_h
        self.flip_v = flip_v

    def __call__(self, tag_data):
        if self.rotate is not None:
            tag_data = np.rot90(tag_data, (360 - self.rotate) / 90)
        if self.flip_h:
            tag_data = np.fliplr(tag_data)
        if self.flip_v:
            tag_data = np.flipud(tag_data)
        return tag_data

    @staticmethod
    def create_from_config(config):
        notify_config = config["notify"]
        rotate = notify_config["rotate"]
        flip_h = notify_config["flipH"]
        flip_v = notify_config["flipV"]
        return TransformTagData(rotate, flip_h, flip_v)
