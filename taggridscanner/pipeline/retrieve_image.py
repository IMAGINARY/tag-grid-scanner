from abc import ABCMeta, abstractmethod
import time
import cv2
import numpy as np

from taggridscanner.aux.threading import WorkerThread, SynchronizedObjectProxy


class RetrieveImage(metaclass=ABCMeta):
    def __init__(self, capture):
        super().__init__()
        self.capture = capture

    @abstractmethod
    def read(self):
        pass

    @property
    def size(self):
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (h, w)

    def __del(self):
        self.capture.release()

    def __call__(self):
        return self.read()

    @staticmethod
    def create_from_config(config):
        camera_config = config["camera"]
        use_camera_device = "id" in camera_config
        source = camera_config["id"] if use_camera_device else camera_config["filename"]
        capture = cv2.VideoCapture(source)

        if "fourcc" in camera_config:
            s = camera_config["fourcc"]
            fourcc = cv2.VideoWriter_fourcc(s[0], s[1], s[2], s[3])
            capture.set(cv2.CAP_PROP_FOURCC, fourcc)

        if "size" in camera_config:
            [height, width] = camera_config["size"]
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if "fps" in camera_config:
            fps = camera_config["fps"]
            capture.set(cv2.CAP_PROP_FPS, fps)

        if "exposure" in camera_config:
            exposure = camera_config["exposure"]
            capture.set(cv2.CAP_PROP_EXPOSURE, exposure)

        if use_camera_device:
            return RetrieveFromCamera(capture)
        else:
            num_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if num_frames == 1.0:
                # This is just a heuristic. OpenCV's capture API sucks.
                return RetrieveFromSingleImage(capture)
            else:
                return RetrieveFromVideo(capture)


class RetrieveFromCamera(RetrieveImage):
    def __init__(self, capture):
        super().__init__(capture)

    def read(self):
        ret, img = self.capture.read()
        return img if ret else np.zeros(self.size, np.uint8)


class RetrieveFromSingleImage(RetrieveImage):
    def __init__(self, capture):
        super().__init__(capture)
        fps = 60.0
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.__last_read_ts = -1.0 / fps
        ret, img = self.capture.read()
        self.__image = img if ret else np.zeros(self.size, np.uint8)

    def read(self):
        ts = time.perf_counter()
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            sleep_time = 1.0 / fps - (ts - self.__last_read_ts)
            self.__last_read_ts = ts
            if sleep_time > 0:
                time.sleep(sleep_time)
        return self.__image.copy()


class RetrieveFromVideo(RetrieveImage):
    def __init__(self, capture):
        super().__init__(SynchronizedObjectProxy(capture))

        def read():
            ret, img = capture.read()
            if not ret:
                # reach end of stream -> rewind
                # (can also happen when there is an input error,
                # but there is no way in OpenCV to tell the difference)
                # maybe switch to PyAV for capturing
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                capture.set(cv2.CAP_PROP_POS_MSEC, 0.0)

                # try again
                ret, img = self.capture.read()
                return img if ret else np.zeros(self.size, np.uint8)
            else:
                return img

        fps = capture.get(cv2.CAP_PROP_FPS)
        rate_limit = fps if fps > 0.0 else None
        self.worker = WorkerThread(read, rate_limit=rate_limit)
        self.worker.start()

    def __del(self):
        self.worker.stop()

    def read(self):
        return self.worker.result.retrieve()
