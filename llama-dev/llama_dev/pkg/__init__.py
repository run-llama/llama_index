import click

from .bump import bump
from .cmd_exec import cmd_exec
from .info import info


@click.group(short_help="Manage packages in the monorepo")
def pkg():
    pass  # pragma: no cover


pkg.add_command(info)
pkg.add_command(cmd_exec, name="exec")
pkg.add_command(bump)
