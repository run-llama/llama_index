from pathlib import Path

import click
from rich.console import Console
from rich.theme import Theme

from .pkg import pkg
from .test import test

LLAMA_DEV_THEME = Theme(
    {
        "repr.path": "",
        "repr.filename": "",
        "repr.str": "",
        "traceback.note": "cyan",
        "info": "dim cyan",
        "warning": "magenta",
        "error": "bold red",
    }
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
@click.option(
    "--repo-root",
    default=".",
    help="Path to the llama_index repository, defaults to '.'",
)
@click.option("--debug", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(ctx, repo_root: str, debug: bool):
    """The official CLI for development, testing, and automation in the LlamaIndex monorepo."""
    ctx.obj = {
        "console": Console(theme=LLAMA_DEV_THEME, soft_wrap=True),
        "repo_root": Path(repo_root).resolve(),
        "debug": debug,
    }


cli.add_command(pkg)
cli.add_command(test)
