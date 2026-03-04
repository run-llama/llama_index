import click

from .changelog import changelog
from .check import check
from .prepare import prepare


@click.group(short_help="Utilities for the release process in the monorepo")
def release():
    pass  # pragma: no cover


release.add_command(check)
release.add_command(changelog)
release.add_command(prepare)
