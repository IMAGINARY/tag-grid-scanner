import threading
import time
import cv2
import logging


from taggridscanner.aux.threading import WorkerThreadWithResult


logger = logging.getLogger(__name__)


class WrongImageSizeError(Exception):
    def __init__(self, expected_size, actual_size):
        message = f"Expected image size {expected_size}, but got {actual_size}."
        super().__init__(message)
        self.expected_size = expected_size
        self.actual_size = actual_size


class RetrieveImage:
    def __init__(
        self,
        id_or_filename,
        api_preference=cv2.CAP_ANY,
        props=None,
        reconnection_delay=0.5,
    ):
        super().__init__()
        if props is None:
            props = []
        self.id_or_filename = id_or_filename

        self.api_preference = api_preference
        self.reconnection_delay = reconnection_delay
        self.props = props
        self.capture = cv2.VideoCapture()

        def find_last_prop_value(prop_list, prop_id, default):
            return next((v for (k, v) in reversed(props) if k == prop_id), default)

        default_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        width = find_last_prop_value(self.props, cv2.CAP_PROP_FRAME_WIDTH, default_width)

        default_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        height = find_last_prop_value(self.props, cv2.CAP_PROP_FRAME_HEIGHT, default_height)

        self.__size = (int(height), int(width))

        self.__last_reconnection_ts = float("-inf")
        self.__rlock = threading.RLock()
        self.__last_image = self.__read_and_block_when_disconnected()
        self.__worker = WorkerThreadWithResult(lambda: self.__read())
        self.__worker.start()

    @property
    def size(self):
        return tuple(self.__size)

    @property
    def rlock(self):
        return self.__rlock

    def read(self):
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.__worker.rate_limit = 60.0 if fps == 0.0 else fps
        return self.__worker.result.retrieve()

    def __read_and_block_when_disconnected(self):
        with self.rlock:
            ret, image = self.capture.read()
            while not ret:
                self.reconnect()
                ret, image = self.capture.read()

            image_size = (image.shape[0], image.shape[1])
            if image_size != self.size:
                m = f"Camera image size {image_size} does not match the expected size {self.size}. Aborting."
                logger.error(m)
                raise WrongImageSizeError(self.size, image_size)
            else:
                self.__last_image = image

            return image

    def __read(self):
        with self.rlock:
            ret, image = self.capture.read()
            if not ret:
                self.reconnect()
                ret, image = self.capture.read()

            if ret:
                image_size = (image.shape[0], image.shape[1])
                if image_size != self.size:
                    m = f"Camera image size {image_size} does not match the expected size {self.size}. Skipping frame."
                    logger.warning(m)
                else:
                    self.__last_image = image

            return self.__last_image

    def reconnect(self):
        with self.rlock:
            ret = False
            while not ret:
                reconnection_ts = time.perf_counter()
                reconnection_delay = self.reconnection_delay - (reconnection_ts - self.__last_reconnection_ts)
                self.__last_reconnection_ts = reconnection_ts
                if reconnection_delay > 0:
                    time.sleep(reconnection_delay)

                self.capture.release()
                ret = self.capture.open(self.id_or_filename, self.api_preference)
                if ret:
                    for prop_id, prop_value in self.props:
                        self.capture.set(prop_id, prop_value)

    @property
    def size(self):
        return tuple(self.__size)

    def __del(self):
        with self.rlock:
            self.__worker.stop()
            self.capture.release()

    def __call__(self):
        return self.read()

    @staticmethod
    def create_from_config(config):
        camera_config = config["camera"]
        use_camera_device = "id" in camera_config
        source = camera_config["id"] if use_camera_device else camera_config["filename"]

        props = []
        if "fourcc" in camera_config:
            s = camera_config["fourcc"]
            fourcc = cv2.VideoWriter_fourcc(s[0], s[1], s[2], s[3])
            props.append((cv2.CAP_PROP_FOURCC, fourcc))

        if "size" in camera_config:
            [height, width] = camera_config["size"]
            props.append((cv2.CAP_PROP_FRAME_WIDTH, width))
            props.append((cv2.CAP_PROP_FRAME_HEIGHT, height))

        if "fps" in camera_config:
            fps = camera_config["fps"]
            props.append((cv2.CAP_PROP_FPS, fps))

        return RetrieveImage(
            source,
            props=props,
        )
