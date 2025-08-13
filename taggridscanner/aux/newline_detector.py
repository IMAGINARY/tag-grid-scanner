import threading
from taggridscanner.aux.threading import WorkerThreadWithResult


def input_catch_eof():
    try:
        return input()
    except EOFError:
        threading.Event().wait()


class NewlineDetector(WorkerThreadWithResult):
    def __init__(self):
        super().__init__(input_catch_eof)
