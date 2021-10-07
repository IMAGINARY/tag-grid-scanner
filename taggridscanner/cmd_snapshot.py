import sys
import cv2

from taggridscanner.utils import setup_video_capture


def snapshot(args):
    config = args["config"]
    config_with_defaults = args["config-with-defaults"]
    camera_config = config_with_defaults["camera"]

    capture = setup_video_capture(camera_config)

    while True:
        ret, frame = capture.read()

        if ret:
            cv2.imshow("snapshot", frame)

            key = cv2.waitKey(1)
            if key == 32 or key == 13:
                if "filename" in camera_config:
                    print(
                        "Saving snapshot to {}".format(config["camera"]["filename"]),
                        file=sys.stderr,
                    )
                    cv2.imwrite(camera_config["filename"], frame)
                else:
                    print(
                        'config["camera"]["filename"] not specified. Not saving.',
                        file=sys.stderr,
                    )
                key = cv2.waitKey(1000)
            elif key == 27:
                break
        else:
            key = cv2.waitKey(1)
            if key == 27:
                break

    capture.release()
    cv2.destroyAllWindows()
