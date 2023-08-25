from .configuration import load_config, save_config
from argparse import Namespace, _SubParsersAction


def init_cli(args: Namespace) -> None:
    """Handle subcommand "init" """
    config = load_config(args.directory)
    save_config(config, args.directory)


def register_init_cli(subparsers: _SubParsersAction) -> None:
    """Register subcommand "init" to ArgumentParser"""
    parser = subparsers.add_parser("init")
    parser.add_argument(
        "directory",
        default=".",
        nargs="?",
        help="Directory to init",
    )

    parser.set_defaults(func=init_cli)
