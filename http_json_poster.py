import json
import threading
import requests


class HttpJsonPoster:
    __headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def __init__(self, url, interval=None):
        self.__cond = threading.Condition()
        self.__url = url
        self.__last_json = None
        self.__new_json = None
        self.__interval = interval
        self.__thread = threading.Thread(target=lambda: self.__loop(), daemon=True)
        self.__thread.start()

        # init the thread

    @property
    def interval(self):
        return self.__interval

    @interval.setter
    def interval(self, interval):
        with self.__cond:
            self.__interval = interval
            self.__cond.notifyAll()

    @property
    def url(self):
        return self.__url

    @url.setter
    def url(self, url):
        with self.__cond:
            self.__url = url
            self.__cond.notifyAll()

    def request_post(self, value):
        with self.__cond:
            self.__new_json = json.dumps(value).encode("utf-8")
            self.__cond.notifyAll()

    def __loop(self):
        while True:
            with self.__cond:
                if self.__new_json is None:
                    self.__cond.wait(self.__interval)

                # if there is not data: overwrite the old data
                if self.__new_json is not None:
                    self.__last_json = self.__new_json
                    self.__new_json = None

            # send the most recent data
            if self.__last_json is not None:
                self.__post()

    def __post(self):
        url = self.__url
        r = requests.post(url, data=self.__last_json, headers=HttpJsonPoster.__headers)
        if not r.status_code == 200:
            msg = "Error sending request:"
            print(msg, r.status_code, r.reason, url, self.__last_json)
