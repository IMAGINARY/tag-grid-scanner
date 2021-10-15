import sys
from taggridscanner.aux.threading import WorkerThread


class NewlineDetector(WorkerThread):
    def __init__(self):
        super().__init__(input)
