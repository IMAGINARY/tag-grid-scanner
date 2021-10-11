import uuid
import cv2
from taggridscanner.utils import Functor


class ViewImage(Functor):
    def __init__(self, title=None):
        super().__init__()
        self.__window_name = str(uuid.uuid4())
        cv2.namedWindow(self.__window_name)
        self.__title = self.__window_name if title is None else title
        self.title = title

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, title):
        self.__title = title
        cv2.setWindowTitle(self.__window_name, title)

    def __call__(self, value):
        cv2.imshow(self.__window_name, value)
        return value

    def __del__(self):
        cv2.destroyWindow(self.__window_name)
