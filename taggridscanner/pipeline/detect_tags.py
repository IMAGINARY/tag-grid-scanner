import numpy as np

from taggridscanner.aux.utils import Functor, split_image


class DetectTags(Functor):
    def __init__(self, grid_shape, tag_shape, tags, detect_rotations=True):
        super().__init__()
        self.grid_shape = grid_shape
        self.__tag_shape = tag_shape
        self.__tags = None
        self.__detect_rotations = detect_rotations
        self.__tag_dict, self.__data_for_unknown_tag = self.create_tag_dict(tags)

    @property
    def tag_shape(self):
        return self.__tag_shape

    @tag_shape.setter
    def tag_shape(self, s):
        self.__tag_shape = s
        self.__tag_dict, self.__data_for_unknown_tag = self.create_tag_dict(self.__tags)

    @property
    def tags(self):
        return self.__tags

    @tags.setter
    def tags(self, t):
        self.__tags = t
        self.__tag_dict, self.__data_for_unknown_tag = self.create_tag_dict(self.__tags)

    @property
    def detect_rotations(self):
        return self.__detect_rotations

    @detect_rotations.setter
    def detect_rotations(self, r):
        self.__detect_rotations = r
        self.__tag_dict, self.__data_for_unknown_tag = self.create_tag_dict(self.__tags)

    def detect_tag(self, tile):
        return self.__tag_dict.get(tile.tobytes(), self.__data_for_unknown_tag)

    def create_empty_tags(self):
        return np.full(self.grid_shape, self.__data_for_unknown_tag, dtype=object)

    def create_tag_dict(self, string_tags):
        data_for_unknown_tag = None
        tag_dict = {}
        for string_tag, data in string_tags.items():
            if string_tag == "unknown":
                data_for_unknown_tag = data
            else:
                np_tag = string_tag_to_np_tag(string_tag, self.tag_shape)
                tag_dict[np_tag.tobytes()] = data
                if self.detect_rotations:
                    np_tag = np.rot90(np_tag)
                    tag_dict[np_tag.tobytes()] = data
                    np_tag = np.rot90(np_tag)
                    tag_dict[np_tag.tobytes()] = data
                    np_tag = np.rot90(np_tag)
                    tag_dict[np_tag.tobytes()] = data
        return (tag_dict, data_for_unknown_tag)

    def __call__(self, image):
        assert (
            image.shape[0] == self.grid_shape[0] * self.tag_shape[0]
            and image.shape[1] == self.grid_shape[1] * self.tag_shape[1]
            and len(image.shape) == 2
            and image.dtype == np.uint8
        )

        binary_image = image // 255

        tiles = split_image(binary_image, self.grid_shape, self.tag_shape)

        detected_tags = self.create_empty_tags()
        for y in range(0, tiles.shape[0]):
            for x in range(0, tiles.shape[1]):
                detected_tags[y, x] = self.detect_tag(tiles[y, x])

        return detected_tags


def string_tag_to_np_tag(string_tag, tag_shape):
    return np.fromstring(
        ",".join(list(string_tag)), dtype=np.uint8, count=tag_shape[0] * tag_shape[1], sep=","
    ).reshape(tag_shape)


def np_tag_to_string_tag(np_tag, tag_shape):
    return "".join(str(e) for e in list(np_tag.reshape(tag_shape[0] * tag_shape[1])))
