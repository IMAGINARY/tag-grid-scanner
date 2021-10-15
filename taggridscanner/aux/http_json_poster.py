import threading
import requests


class HttpJsonPoster:
    __headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def __init__(self, url):
        self.__cond = threading.Condition()
        self.__url = url
        self.__last_data = None
        self.__new_data = None
        self.__thread = threading.Thread(target=lambda: self.__loop(), daemon=True)
        self.__thread.start()

    @property
    def url(self):
        return self.__url

    @url.setter
    def url(self, url):
        with self.__cond:
            self.__url = url
            self.__cond.notifyAll()

    def request_post(self, data):
        with self.__cond:
            self.__new_data = data.encode("utf-8")
            self.__cond.notifyAll()

    def __loop(self):
        while True:
            with self.__cond:
                if self.__new_data is None:
                    self.__cond.wait()

                # if there is not data: overwrite the old data
                if self.__new_data is not None:
                    self.__last_data = self.__new_data
                    self.__new_data = None

            # send the most recent data
            if self.__last_data is not None:
                self.__post()

    def __post(self):
        url = self.__url
        r = requests.post(url, data=self.__last_data, headers=HttpJsonPoster.__headers)
        if not r.status_code == 200:
            msg = "Error sending request:"
            print(msg, r.status_code, r.reason, url, self.__last_data)
