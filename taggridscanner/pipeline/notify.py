import json
import sys
import jsonpointer
import re
import uuid
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

        self.uuid = str(uuid.uuid4())

        tmp_template = jsonpointer.set_pointer(template, assign_to, "", False)

        # Ensure that the UUID isn't already part of the template (very unlikely, but possible)
        while self.uuid in tmp_template:
            self.uuid = str(uuid.uuid4())

        template_with_uuid = jsonpointer.set_pointer(template, assign_to, self.uuid, False)
        self.string_template_with_uuid = json.dumps(template_with_uuid, indent=2)

        # Extract the line containing the UUID using a regex and use the length of the prefix to determine the
        # indentation. This allows the notification to be formatted correctly in the output.
        print(self.string_template_with_uuid)
        prefix = re.search(r'^( *).*"{uuid}"'.format(uuid=self.uuid), self.string_template_with_uuid, re.MULTILINE)
        self.indent = prefix.group(1)

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

        tag_data_list_string = (
            "[\n{indent}  ".format(indent=self.indent)
            + ",\n{indent}  ".format(indent=self.indent).join([json.dumps(sub_list) for sub_list in tag_data_list])
            + "\n{indent}]".format(indent=self.indent)
        )

        notification = self.string_template_with_uuid.replace('"{uuid}"'.format(uuid=self.uuid), tag_data_list_string)

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
