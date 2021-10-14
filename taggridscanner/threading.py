import threading
import time


class ThreadSafeContainer(object):
    class Empty(Exception):
        def __init__(self):
            super().__init__()

    def __init__(self, content=None, init_content=False):
        super().__init__()
        self.__content = content if init_content else None
        self.__is_full = init_content
        self.__condition = threading.Condition()

    def is_empty(self):
        return not self.__is_full

    def is_full(self):
        return self.__is_full

    def get(self):
        with self.__condition:
            if self.is_empty():
                self.__condition.wait()
            content = self.__content
            self.__is_full = False
            self.__content = None
            return content

    def wait(self):
        with self.__condition:
            if self.__is_full:
                pass
            else:
                self.__condition.wait()

    def get_nowait(self):
        with self.__condition:
            if self.__is_full:
                content = self.__content
                self.__is_full = False
                self.__content = None
                return content
            else:
                raise ThreadSafeContainer.Empty

    def set(self, content):
        with self.__condition:
            self.__is_full = True
            self.__content = content
            self.__condition.notify()


class WorkerThread(object):
    def __init__(self, func, rate_limit=None, daemon=True):
        super().__init__()
        self.__func = func
        self.__result = ThreadSafeContainer()
        self.rate_limit = rate_limit
        self.__thread_lock = threading.RLock()
        self.__thread = None
        self.__is_daemon = daemon
        self.__should_stop = False

    @property
    def result(self):
        return self.__result

    @property
    def is_daemon(self):
        return self.__is_daemon

    @property
    def is_running(self):
        with self.__thread_lock:
            return False if self.__thread is None else self.__thread.is_alive()

    def start(self):
        with self.__thread_lock:
            if self.__thread is None:
                self.__should_stop = False
                self.__thread = threading.Thread(
                    target=lambda: self.run(), daemon=self.is_daemon
                )
                self.__thread.start()

    def stop(self):
        with self.__thread_lock:
            if self.__thread is not None:
                self.__should_stop = True
                self.__thread.join()
                self.__thread = None

    def run(self):
        while not self.__should_stop:
            start_ts = time.perf_counter()
            self.result.set(self.__func())
            end_ts = time.perf_counter()
            duration = end_ts - start_ts
            rate_limit = self.rate_limit
            if isinstance(rate_limit, int) or isinstance(rate_limit, float):
                time.sleep(max(0.0, 1.0 / rate_limit - duration))
            time.sleep(0)
