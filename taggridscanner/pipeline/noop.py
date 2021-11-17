from taggridscanner.aux.utils import Functor


class Noop(Functor):
    def __init__(self):
        super().__init__(lambda img: img)
