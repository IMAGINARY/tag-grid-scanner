import json
import sys
import jsonpointer
import numpy as np

from taggridscanner.aux.http_json_poster import HttpJsonPoster
from taggridscanner.aux.notification_manager import NotificationManager
from taggridscanner.aux.utils import Functor


class Notify(Functor):
    def __init__(
        self,
        template,
        assign_to,
        stdout=True,
        stderr=False,
        url=None,
        interval=None,
    ):
        super().__init__()
        self.template = template
        self.assign_to = assign_to

        notifiers = []
        if stdout:
            notifiers.append(lambda s: print(s, file=sys.stdout))
        if stderr:
            notifiers.append(lambda s: print(s, file=sys.stderr))
        if url is not None:
            http_json_poster = HttpJsonPoster(url)
            notifiers.append(lambda s: http_json_poster.request_post(s))

        self.notification_manager = NotificationManager(notifiers, interval)

    def __call__(self, tag_data):
        tag_data_list = np.array(tag_data).tolist()
        notification_obj = jsonpointer.set_pointer(self.template, self.assign_to, tag_data_list, False)
        notification = json.dumps(notification_obj)
        self.notification_manager.notify(notification)
        return tag_data

    @staticmethod
    def create_from_config(config):
        notify_config = config["notify"]
        stdout = notify_config["stdout"]
        stderr = notify_config["stderr"]
        url = notify_config["url"] if notify_config["remote"] else None
        interval = notify_config["interval"] if notify_config["repeat"] else None
        template = notify_config["template"]
        assign_to = notify_config["assignTo"]
        return Notify(template, assign_to, stdout, stderr, url, interval)
