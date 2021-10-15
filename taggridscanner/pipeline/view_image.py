import uuid
import cv2
from taggridscanner.aux.utils import Functor


class ViewImage(Functor):
    def __init__(self, title=None):
        super().__init__()
        self.__window_name = str(uuid.uuid4())
        self.__title = ""
        self.title = self.__window_name if title is None else title

    @property
    def window_name(self):
        return self.__window_name

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, title):
        self.__title = title
        cv2.namedWindow(self.__window_name)
        cv2.setWindowTitle(self.__window_name, title)

    def hide(self):
        cv2.destroyWindow(self.__window_name)

    def __call__(self, value):
        cv2.namedWindow(self.__window_name)
        cv2.setWindowTitle(self.__window_name, self.__title)
        cv2.imshow(self.__window_name, value)
        return value

    def __del__(self):
        self.hide()
