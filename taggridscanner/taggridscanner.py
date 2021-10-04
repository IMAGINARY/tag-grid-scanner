from .arguments import get_arguments
from .config import get_config
from .cmd_scan import scan
from .cmd_display import display
from .cmd_calibrate import calibrate
from .cmd_snapshot import snapshot
from .cmd_roi import roi


def main():
    args = get_arguments()
    config, config_with_defaults = get_config(args["CONFIG_FILE"][0])

    commands = {
        "scan": scan,
        "display": display,
        "calibrate": calibrate,
        "snapshot": snapshot,
        "roi": roi,
    }

    assert args["COMMAND"] in commands

    command = commands[args["COMMAND"]]
    command(args, config, config_with_defaults)
