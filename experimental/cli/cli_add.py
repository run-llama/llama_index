import os
from .configuration import load_index, save_index
from argparse import Namespace, _SubParsersAction
from llama_index import SimpleDirectoryReader


def add_cli(args: Namespace) -> None:
    """Handle subcommand "add" """
    index = load_index()

    for p in args.files:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        if os.path.isdir(p):
            documents = SimpleDirectoryReader(p).load_data()
            for document in documents:
                index.insert(document)
        else:
            documents = SimpleDirectoryReader(input_files=[p]).load_data()
            for document in documents:
                index.insert(document)

    save_index(index)


def register_add_cli(subparsers: _SubParsersAction) -> None:
    """Register subcommand "add" to ArgumentParser"""
    parser = subparsers.add_parser("add")
    parser.add_argument(
        "files",
        default=".",
        nargs="+",
        help="Files to add",
    )

    parser.set_defaults(func=add_cli)
