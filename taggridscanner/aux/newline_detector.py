import sys
from taggridscanner.aux.threading import WorkerThreadWithResult


class NewlineDetector(WorkerThreadWithResult):
    def __init__(self):
        super().__init__(input)
