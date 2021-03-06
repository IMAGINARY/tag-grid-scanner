import threading
import time
from wrapt import synchronized, ObjectProxy


@synchronized
class SynchronizedObjectProxy(ObjectProxy):
    pass


class ThreadSafeValue(object):
    def __init__(self, initial_value):
        super().__init__()
        self.__value = initial_value
        self.__condition = threading.Condition()

    @property
    def condition(self):
        return self.__condition

    def get(self):
        with self.__condition:
            return self.__value

    def get_nowait(self):
        return self.__value

    def set(self, value):
        with self.__condition:
            self.__value = value
            self.__condition.notify()


class ThreadSafeContainer(object):
    class Empty(Exception):
        def __init__(self):
            super().__init__()

    def __init__(self, *args):
        super().__init__()
        self.__content, self.__is_full = (
            (args[0], True) if len(args) > 0 else (None, False)
        )
        self.__condition = threading.Condition()

    @property
    def condition(self):
        return self.__condition

    def is_empty(self):
        return not self.__is_full

    def is_full(self):
        return self.__is_full

    def wait(self):
        with self.__condition:
            if self.__is_full:
                pass
            else:
                self.__condition.wait()

    def get(self):
        with self.__condition:
            if self.is_empty():
                self.wait()
            return self.__content

    def get_nowait(self):
        with self.__condition:
            if self.is_full():
                return self.__content
            else:
                raise ThreadSafeContainer.Empty

    def retrieve(self):
        with self.__condition:
            if self.is_empty():
                self.wait()
            content = self.__content
            self.__is_full = False
            self.__content = None
            return content

    def retrieve_nowait(self):
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
        self.rate_limit = rate_limit
        self.__thread_lock = threading.RLock()
        self.__thread = None
        self.__is_daemon = daemon
        self.__should_stop = False

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
            self.__func()
            end_ts = time.perf_counter()
            duration = end_ts - start_ts
            rate_limit = self.rate_limit
            if isinstance(rate_limit, int) or isinstance(rate_limit, float):
                time.sleep(max(0.0, 1.0 / rate_limit - duration))
            time.sleep(0)


class WorkerThreadWithResult(WorkerThread):
    def __init__(self, func, rate_limit=None, daemon=True):
        super().__init__(lambda: self.result.set(func()), rate_limit, daemon)
        self.__result = ThreadSafeContainer()

    @property
    def result(self):
        return self.__result
