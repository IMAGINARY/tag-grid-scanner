import sys
import cv2

from taggridscanner.utils import setup_video_capture


def snapshot(args):
    config = args["config"]
    config_with_defaults = args["config-with-defaults"]
    camera_config = config_with_defaults["camera"]
    output_filename = args.get("OUTFILE", None)

    capture = setup_video_capture(camera_config)

    while True:
        ret, frame = capture.read()

        if ret:
            cv2.imshow("snapshot", frame)

            key = cv2.waitKey(1)
            if key == 32 or key == 13:
                if output_filename is not None:
                    print(
                        "Saving snapshot to {}".format(output_filename),
                        file=sys.stderr,
                    )
                    cv2.imwrite(output_filename, frame)
                else:
                    print(
                        "No output filename specified. Not saving.",
                        file=sys.stderr,
                    )
                cv2.waitKey(1000)
            elif key == 27:
                break
        else:
            key = cv2.waitKey(1)
            if key == 27:
                break

    capture.release()
    cv2.destroyAllWindows()
