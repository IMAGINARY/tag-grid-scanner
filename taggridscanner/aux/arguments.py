import argparse
import pathlib

from taggridscanner.aux.config import load_config
from taggridscanner.aux.logging import configure_logging
from taggridscanner.cmd.scan import scan
from taggridscanner.cmd.display import display
from taggridscanner.cmd.calibrate import calibrate
from taggridscanner.cmd.snapshot import snapshot


class ConfigParseAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        config, config_with_defaults, raw_config = load_config(values)
        setattr(namespace, "config-path", values)
        setattr(namespace, "config", config)
        setattr(namespace, "config-with-defaults", config_with_defaults)
        setattr(namespace, "raw-config", raw_config)


def add_config_argument(parser):
    return parser.add_argument(
        "-c",
        "--config",
        type=pathlib.Path,
        nargs=1,
        required=True,
        action=ConfigParseAction,
        help="configuration file to load",
    )


def get_argument_parser():
    parser = argparse.ArgumentParser(prog="tag-grid-scanner")
    sub_parsers = parser.add_subparsers(help="run the given command", required=True)

    # parser for 'calibrate' command
    parser_calibrate = sub_parsers.add_parser(
        "calibrate",
        help="compute the camera matrix and distortion coefficients for a specific combination of camera and lens "
        "(given a fixed focal length and focus)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_calibrate.set_defaults(func=calibrate)
    parser_calibrate.add_argument(
        "--width",
        type=int,
        default=1920,
        help="width of calibration pattern image",
    )
    parser_calibrate.add_argument(
        "--height",
        type=int,
        default=1080,
        help="height of calibration pattern image",
    )
    parser_calibrate.add_argument(
        "--rows",
        type=int,
        default=17,
        help="rows of the checkerboard calibration pattern",
    )
    parser_calibrate.add_argument(
        "--cols",
        type=int,
        default=31,
        help="columns of the checkerboard calibration pattern",
    )
    parser_calibrate.add_argument(
        "-n",
        type=int,
        default=10,
        help="number of pictures taken",
    )
    parser_calibrate.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.25,
        help="per frame calibration error tolerance",
    )
    parser_calibrate.add_argument(
        "--no-pattern",
        action="store_true",
        help="do not show the calibration pattern",
    )

    # parser for 'display' command
    parser_display = sub_parsers.add_parser(
        "display",
        help="display random grids of random tags (for assisting development)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_display.set_defaults(func=display)

    # parser for 'scan' command
    parser_scan = sub_parsers.add_parser(
        "scan",
        help="launch ROI editor, detect tags and notify",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_scan.set_defaults(func=scan)
    parser_scan.add_argument(
        "--auto-hide-gui",
        type=float,
        default=float("inf"),
        metavar="SECONDS",
        help="automatically hide all windows when no key is pressed for the given number of seconds",
    )
    parser_scan.add_argument(
        "--hide-gui",
        action="store_true",
        help="start with hidden windows",
    )
    parser_scan.add_argument(
        "--no-gui",
        action="store_true",
        help="disable all windows for headless operation",
    )
    parser_scan.add_argument(
        "--rate-limit",
        type=float,
        default=4,
        metavar="FPS",
        help="limit detection rate to save resources",
    )
    parser_scan.add_argument(
        "--ignore-scale",
        action="store_true",
        help="ignore the scale config option, e.g. to improve ROI selection",
    )
    parser_scan.add_argument(
        "--no-notify",
        action="store_true",
        help="do not send or output any tag data notification",
    )

    # parser for 'snapshot' command
    parser_snapshot = sub_parsers.add_parser(
        "snapshot",
        help="take a snapshot image (for assisting development)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_snapshot.set_defaults(func=snapshot)
    parser_snapshot.add_argument(
        "--headless",
        action="store_true",
        help="do not show the video stream",
    )
    parser_snapshot.add_argument(
        "--wait",
        type=float,
        metavar="WAIT_SEC",
        help="wait for the given number of seconds, take a snapshot and exit",
    )
    parser_snapshot.add_argument(
        "--roi",
        action="store_true",
        help="preprocess the image and draw the region of interest",
    )
    parser_snapshot.add_argument(
        "OUTFILE", nargs="?", help="file to store the snapshot"
    )

    all_subparsers = [
        parser_calibrate,
        parser_display,
        parser_scan,
        parser_snapshot,
    ]
    for sub_parser in all_subparsers:
        sub_parser.add_argument(
            "-c",
            "--config",
            type=pathlib.Path,
            required=True,
            action=ConfigParseAction,
            help="configuration file to load",
        )
        sub_parser.add_argument(
            "-v",
            "--verbose",
            action='count',
            default=3,
            help="increase verbosity (can be used multiple times)",
        )
        sub_parser.add_argument(
            "-q",
            "--quiet",
            action='count',
            default=0,
            help="decrease verbosity (can be used multiple times)",
        )

    return parser


def process_arguments():
    parser = get_argument_parser()
    args = vars(parser.parse_args())

    verbosity = args["verbose"] - args["quiet"]
    del args["verbose"]
    del args["quiet"]
    configure_logging(verbosity)

    func = args["func"]
    del args["func"]
    func(args)
