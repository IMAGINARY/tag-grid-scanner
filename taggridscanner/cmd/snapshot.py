import sys
import cv2
from time import sleep
from taggridscanner.aux.newline_detector import NewlineDetector
from taggridscanner.aux.threading import WorkerThread, ThreadSafeContainer
from taggridscanner.aux.utils import Timeout
from taggridscanner.pipeline.retrieve_image import RetrieveImage
from taggridscanner.pipeline.view_image import ViewImage


def with_ui(args, retrieve_image_worker, output_filename):
    frame = retrieve_image_worker.result.retrieve()
    view_image = ViewImage("Snapshot")

    if args["wait"] is not None:
        auto_snap_timeout = Timeout(args["wait"])
        print("Will take a snapshot after {}s".format(args["wait"]), flush=True)
    else:
        auto_snap_timeout = Timeout(float("inf"))

    key = 0
    while key != 27 and key != ord("q"):
        print(
            "Press SPACE or ENTER to take a snapshot. Press ESC or q to quit.",
            file=sys.stderr,
        )

        while key != 27 and key != ord("q"):
            try:
                frame = retrieve_image_worker.result.retrieve_nowait()
                view_image(frame)
            except ThreadSafeContainer.Empty:
                pass
            key = cv2.waitKey(1)
            if auto_snap_timeout.is_up() or key == 32 or key == 13:
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
                if auto_snap_timeout.is_up():
                    key = 27
                    break

                print("Press ESC or q to quit. Press any other key to try again.")
                key = cv2.waitKey()
                break


def headless(args, retrieve_image_worker, output_filename):
    newline_detector = NewlineDetector()
    newline_detector.start()

    if args["wait"] is not None:
        print("Waiting {}s ...".format(args["wait"]), end="", flush=True)
        sleep(args["wait"])
        print("done", flush=True)
        frame = retrieve_image_worker.result.retrieve()
        cv2.imwrite(output_filename, frame)
    else:
        while True:
            print("Press ENTER to take a snapshot. Abort with Ctrl+C.", file=sys.stderr)
            newline_detector.result.retrieve()
            print(
                "Saving snapshot to {}".format(output_filename),
                file=sys.stderr,
            )
            frame = retrieve_image_worker.result.retrieve()
            cv2.imwrite(output_filename, frame)


def snapshot(args):
    config_with_defaults = args["config-with-defaults"]
    output_filename = args.get("OUTFILE", None)

    retrieve_image = RetrieveImage.create_from_config(config_with_defaults)
    retrieve_image.scale = (1.0, 1.0)
    retrieve_image_worker = WorkerThread(retrieve_image)

    retrieve_image_worker.start()

    if args["headless"]:
        headless(args, retrieve_image_worker, output_filename)
    else:
        with_ui(args, retrieve_image_worker, output_filename)
