import argparse
import pathlib


def get_argument_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "COMMAND",
        choices=["scan", "display"],
        const="scan",
        default="scan",
        nargs="?",
        help="command to execute",
    )
    parser.add_argument(
        "CONFIG_FILE",
        type=pathlib.Path,
        nargs=1,
        help="path to the JSON configuration file",
    )
    return parser


def get_arguments():
    parser = get_argument_parser()
    args = vars(parser.parse_args())
    return args
