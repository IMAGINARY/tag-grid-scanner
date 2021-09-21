from .arguments import get_arguments
from .config import get_config
from .cmd_scan import scan


def main():
    args = get_arguments()
    config, config_with_defaults = get_config(args["CONFIG_FILE"][0])

    commands = {
        "scan": scan,
    }

    assert args["COMMAND"] in commands

    command = commands[args["COMMAND"]]
    command(args, config, config_with_defaults)
