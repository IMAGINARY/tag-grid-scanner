import uuid
import cv2
from taggridscanner.aux.utils import Functor


class ViewImage(Functor):
    def __init__(self, title=None):
        super().__init__()
        self.__window_name = str(uuid.uuid4())
        self.__window_flags = (
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
        )
        self.__title = ""
        self.title = self.__window_name if title is None else title
        self.__last_image_shape = (0, 0)

    @property
    def window_name(self):
        return self.__window_name

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, title):
        self.__title = title
        cv2.namedWindow(self.__window_name, self.__window_flags)
        cv2.setWindowTitle(self.__window_name, title)

    def hide(self):
        cv2.destroyWindow(self.__window_name)

    def __call__(self, image):
        cv2.namedWindow(self.__window_name, self.__window_flags)
        cv2.setWindowTitle(self.__window_name, self.__title)
        if self.__last_image_shape != image.shape[0:2]:
            cv2.resizeWindow(self.__window_name, image.shape[1], image.shape[0])
            self.__last_image_shape = image.shape[0:2]
        cv2.imshow(self.__window_name, image)
        return image

    def __del__(self):
        self.hide()
