import logging
from argparse import ArgumentParser, Namespace
from .cli_init import register_init_cli
from .cli_add import register_add_cli
from .cli_query import register_query_cli

logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description=None)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s " + "1.0",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="Set verbosity.",
    )

    def print_help(kwargs: Namespace) -> None:
        parser.print_help()

    subparsers = parser.add_subparsers()
    register_init_cli(subparsers)
    register_add_cli(subparsers)
    register_query_cli(subparsers)
    parser.set_defaults(func=print_help)

    args = parser.parse_args()
    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 2:
        logger.setLevel(logging.DEBUG)

    args.func(args)


if __name__ == "__main__":
    main()
