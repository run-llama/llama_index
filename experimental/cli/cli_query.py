from .configuration import load_index
from argparse import Namespace, _SubParsersAction


def query_cli(args: Namespace) -> None:
    """Handle subcommand "query" """
    index = load_index()
    query_engine = index.as_query_engine()
    print(query_engine.query(args.query))


def register_query_cli(subparsers: _SubParsersAction) -> None:
    """Register subcommand "query" to ArgumentParser"""
    parser = subparsers.add_parser("query")
    parser.add_argument(
        "query",
        help="Query",
    )

    parser.set_defaults(func=query_cli)
