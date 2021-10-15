import sys
import cv2
from taggridscanner.pipeline.retrieve_image import RetrieveImage
from taggridscanner.pipeline.view_image import ViewImage


def snapshot(args):
    config_with_defaults = args["config-with-defaults"]
    output_filename = args.get("OUTFILE", None)

    retrieve_image = RetrieveImage.create_from_config(config_with_defaults)
    view_image = ViewImage("Snapshot")

    key = 0

    while key != 27 and key != ord("q"):
        print(
            "Press SPACE or ENTER to take a snapshot. Press ESC or q to quit.",
            file=sys.stderr,
        )
        while key != 27 and key != ord("q"):
            frame = retrieve_image()
            view_image(frame)

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
                print("Press ESC or q to quit. Press any other key to try again.")
                key = cv2.waitKey()
                break
