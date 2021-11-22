import threading
import time
import cv2

from taggridscanner.aux.threading import WorkerThreadWithResult
from taggridscanner.aux.utils import compatible


class RetrieveImage:
    def __init__(
        self,
        id_or_filename,
        api_preference=cv2.CAP_ANY,
        props=None,
        reconnection_delay=0.5,
        scale=(1.0, 1.0),
        smooth=0.0,
    ):
        super().__init__()
        if props is None:
            props = []
        self.id_or_filename = id_or_filename
        self.api_preference = api_preference
        self.reconnection_delay = reconnection_delay
        self.__scale = None
        self.__prescale = None
        self.scale = scale  # this also initializes the prescaler
        self.smooth = smooth
        self.props = props
        self.capture = cv2.VideoCapture()
        self.__last_reconnection_ts = float("-inf")
        self.__rlock = threading.RLock()
        self.__last_image = self.__read_and_block_when_disconnected()
        self.__worker = WorkerThreadWithResult(lambda: self.__read())
        self.__worker.start()

    @property
    def rlock(self):
        return self.__rlock

    def __read_from_capture(self):
        ret, image = self.capture.read()
        return ret, self.__prescale(image) if image is not None else image

    def read(self):
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.__worker.rate_limit = 60.0 if fps == 0.0 else fps
        return self.__worker.result.retrieve()

    def __read_and_block_when_disconnected(self):
        with self.rlock:
            ret, image = self.__read_from_capture()
            while not ret:
                self.reconnect()
                ret, image = self.__read_from_capture()
            return image

    def __read(self):
        with self.rlock:
            ret, image = self.__read_from_capture()
            if not ret:
                self.reconnect()
                ret, image = self.__read_from_capture()

            if ret:
                if not compatible(self.__last_image, image) or self.smooth == 0.0:
                    self.__last_image = image
                else:
                    cv2.addWeighted(
                        self.__last_image,
                        self.smooth,
                        image,
                        1.0 - self.smooth,
                        0.0,
                        dst=self.__last_image,
                    )
            return self.__last_image

    def reconnect(self):
        with self.rlock:
            ret = False
            while not ret:
                reconnection_ts = time.perf_counter()
                reconnection_delay = self.reconnection_delay - (
                    reconnection_ts - self.__last_reconnection_ts
                )
                self.__last_reconnection_ts = reconnection_ts
                if reconnection_delay > 0:
                    time.sleep(reconnection_delay)

                self.capture.release()
                ret = self.capture.open(self.id_or_filename, self.api_preference)
                if ret:
                    for (prop_id, prop_value) in self.props:
                        self.capture.set(prop_id, prop_value)

    @property
    def size(self):
        with self.rlock:
            w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (h, w)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        assert len(scale) >= 2
        self.__scale = scale
        self.__prescale = create_prescaler(scale)

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

        scale = camera_config["scale"]

        return RetrieveImage(
            source, props=props, scale=scale, smooth=camera_config["smooth"]
        )


def create_prescaler(scale):
    if scale == [1.0, 1.0]:
        return lambda img: img  # noop
    else:

        def prescaler(img):
            prescaled_shape = (
                round(img.shape[1] * scale[1]),
                round(img.shape[0] * scale[0]),
            )
            return cv2.resize(
                img,
                prescaled_shape,
                interpolation=cv2.INTER_AREA,
            )

        return prescaler
