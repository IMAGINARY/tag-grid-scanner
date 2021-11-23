import threading
from copy import deepcopy


class NotificationManager:
    def __init__(self, notifiers, interval=None):
        self.notifiers = notifiers
        self.__cond = threading.Condition()
        self.__notification = None
        self.__new_notification = None
        self.__interval = interval
        self.__thread = threading.Thread(target=lambda: self.__loop(), daemon=True)
        self.__thread.start()

    @property
    def interval(self):
        return self.__interval

    @interval.setter
    def interval(self, interval):
        with self.__cond:
            self.__interval = interval
            self.__cond.notifyAll()

    def notify(self, notification):
        with self.__cond:
            self.__new_notification = deepcopy(notification)
            self.__cond.notifyAll()

    def __loop(self):
        while True:
            with self.__cond:
                if self.__new_notification is None:
                    self.__cond.wait(self.__interval)

                # if there is not data: overwrite the old data
                if self.__new_notification is not None:
                    self.__notification = self.__new_notification
                    self.__new_notification = None

            # send the most recent data
            if self.__notification is not None:
                for notify in self.notifiers:
                    notify(self.__notification)
