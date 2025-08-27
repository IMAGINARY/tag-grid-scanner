import sys
import threading
import requests


class HttpJsonPoster:
    __headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def __init__(self, url, timeout=3):
        self.__cond = threading.Condition()
        self.__url = url
        self.timeout = timeout
        self.__data = None
        self.__new_data = None
        self.__thread = threading.Thread(target=lambda: self.__loop(), daemon=True)
        self.__thread.start()

    @property
    def condition(self):
        return self.__cond

    @property
    def url(self):
        return self.__url

    @url.setter
    def url(self, url):
        with self.__cond:
            self.__url = url
            self.__cond.notifyAll()

    def request_post(self, data):
        data_utf8 = data.encode("utf-8")
        with self.__cond:
            self.__new_data = data_utf8
            self.__cond.notifyAll()

    def __loop(self):
        while True:
            with self.__cond:
                if self.__new_data is None:
                    self.__cond.wait()

                # if there is new data -> overwrite the old data
                if self.__new_data is not None:
                    self.__data = self.__new_data
                    self.__new_data = None

            # send the most recent data
            if self.__data is not None:
                self.__post()

    def __post(self):
        with self.__cond:
            url = self.url
            timeout = self.timeout
            data = self.__data

        error_msg = "Error sending request:"
        try:
            with requests.post(
                url,
                data=data,
                headers=HttpJsonPoster.__headers,
                timeout=timeout,
            ) as r:
                if not r.status_code == 200:
                    print(error_msg, r.status_code, r.reason, r.text, url, data, file=sys.stderr)
        except requests.RequestException as e:
            print(error_msg, e, url, data, file=sys.stderr)
