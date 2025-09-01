import sys
import cv2
from time import sleep
from taggridscanner.aux.threading import WorkerThreadWithResult, ThreadSafeContainer
from taggridscanner.aux.utils import Timeout
from taggridscanner.pipeline.retrieve_image import RetrieveImage
from taggridscanner.pipeline.view_image import ViewImage
from taggridscanner.cmd.scan import ScanWorker


def with_ui(get_frame, wait, output_filename):
    view_image = ViewImage("Snapshot")

    if wait is not None:
        auto_snap_timeout = Timeout(wait)
        print("Will take a snapshot after {}s".format(wait), flush=True)
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
                frame = get_frame()
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


def headless(get_frame, wait, output_filename):
    if wait is not None:
        print("Waiting {}s ... ".format(wait), end="", flush=True)
        sleep(wait)
        print("done", flush=True)
        frame = get_frame()
        cv2.imwrite(output_filename, frame)
    else:
        while True:
            print("Press ENTER to take a snapshot. Abort with Ctrl+C.", file=sys.stderr)
            try:
                input()
            except KeyboardInterrupt:
                sys.exit(0)
            print(
                "Saving snapshot to {}".format(output_filename),
                file=sys.stderr,
            )
            frame = get_frame()
            cv2.imwrite(output_filename, frame)


def create_src_frame_grabber(config):
    config["camera"]["scale"] = [1.0, 1.0]
    config["camera"]["rotate"] = 0

    retrieve_image = RetrieveImage.create_from_config(config)
    return WorkerThreadWithResult(retrieve_image)


def create_roi_frame_grabber(config):
    config["camera"]["scale"] = [1.0, 1.0]
    config["camera"]["rotate"] = 0
    config["camera"]["flipV"] = False
    config["camera"]["flipH"] = False
    config["camera"]["rotate"] = 0
    config["notify"]["stdout"] = False
    config["notify"]["stderr"] = False
    config["notify"]["remote"] = False
    config["notify"]["repeat"] = False

    scan_worker = ScanWorker(config)

    def compute_roi_viz():
        scan_worker()
        return scan_worker.viz.retrieve()[0]  # roi_editor_img in scan.py

    return WorkerThreadWithResult(compute_roi_viz)


def snapshot(args):
    config = args["config-with-defaults"]

    output_filename = args.get("OUTFILE", None)

    create_frame_grabber = create_src_frame_grabber if not args["roi"] else create_roi_frame_grabber
    frame_grabber = create_frame_grabber(config)
    frame_grabber.start()

    frame = frame_grabber.result.get()

    def get_frame():
        nonlocal frame
        try:
            frame = frame_grabber.result.retrieve_nowait()
        except ThreadSafeContainer.Empty:
            pass
        return frame

    wait = args["wait"]

    if args["headless"]:
        headless(get_frame, wait, output_filename)
    else:
        with_ui(get_frame, wait, output_filename)
