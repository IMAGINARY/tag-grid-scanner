import argparse
import pathlib

from .config import get_config
from .cmd_scan import scan
from .cmd_display import display
from .cmd_calibrate import calibrate
from .cmd_snapshot import snapshot
from .cmd_roi import roi


class ConfigParseAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        config, config_with_defaults = get_config(values[0])
        setattr(namespace, "config-path", values)
        setattr(namespace, "config", config)
        setattr(namespace, "config-with-defaults", config_with_defaults)


def add_config_argument(parser):
    return parser.add_argument(
        "-c",
        "--config",
        type=pathlib.Path,
        nargs=1,
        required=True,
        action=ConfigParseAction,
        help="Load this configuration file.",
    )


def get_argument_parser():
    parser = argparse.ArgumentParser(prog="tag-grid-scanner")
    subparsers = parser.add_subparsers(help="sub-command help", required=True)

    parser_calibrate = subparsers.add_parser("calibrate", help="calibrate command help")
    parser_calibrate.set_defaults(func=calibrate)
    add_config_argument(parser_calibrate)

    parser_display = subparsers.add_parser("display", help="display command help")
    parser_display.set_defaults(func=display)
    add_config_argument(parser_display)

    parser_roi = subparsers.add_parser("roi", help="roi command help")
    parser_roi.set_defaults(func=roi)
    add_config_argument(parser_roi)

    parser_scan = subparsers.add_parser("scan", help="scan command help")
    parser_scan.set_defaults(func=scan)
    add_config_argument(parser_scan)

    parser_snapshot = subparsers.add_parser("snapshot", help="snapshot command help")
    parser_snapshot.set_defaults(func=snapshot)
    add_config_argument(parser_snapshot)

    return parser


def process_arguments():
    parser = get_argument_parser()
    args = vars(parser.parse_args())
    func = args["func"]
    del args["func"]
    func(args)
