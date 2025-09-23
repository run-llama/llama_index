import click

from .check import check


@click.group(short_help="Utilities for the release process in the monorepo")
def release():
    pass  # pragma: no cover


release.add_command(check)
