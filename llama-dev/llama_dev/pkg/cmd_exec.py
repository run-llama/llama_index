import os
import subprocess
from pathlib import Path

import click

from llama_dev.utils import find_all_packages, is_llama_index_package


@click.command(short_help="Exec a command inside a package folder")
@click.option(
    "--fail-fast",
    is_flag=True,
    default=False,
    help="Exit the command at the first failure",
)
@click.option(
    "--all",
    is_flag=True,
    help="Get info for all the packages in the monorepo",
)
@click.argument("package_names", required=False, nargs=-1)
@click.option(
    "--cmd",
    required=True,
    help="The command to execute (use quotes around the full command)",
)
@click.option(
    "--silent",
    is_flag=True,
    default=False,
    help="Only print errors",
)
@click.pass_obj
def cmd_exec(
    obj: dict, all: bool, package_names: tuple, cmd: str, fail_fast: bool, silent: bool
):
    if not all and not package_names:
        raise click.UsageError("Either specify a package name or use the --all flag")

    console = obj["console"]
    packages: set[Path] = set()
    # Do not use the virtual environment calling llama-dev, if any
    env = os.environ.copy()
    if "VIRTUAL_ENV" in env:
        del env["VIRTUAL_ENV"]

    if all:
        packages = set(find_all_packages(obj["repo_root"]))
    else:
        for package_name in package_names:
            package_path = obj["repo_root"] / package_name
            if not is_llama_index_package(package_path):
                raise click.UsageError(
                    f"{package_name} is not a path to a LlamaIndex package"
                )
            packages.add(package_path)

    with console.status(f"[bold green]Running '{cmd}'...") as status:
        for package in packages:
            result = subprocess.run(
                cmd.split(" "),
                cwd=package,
                text=True,
                capture_output=True,
                env=env,
            )
            if result.returncode != 0:
                msg = f"Command '{cmd}' failed in {package.relative_to(obj['repo_root'])}: {result.stderr}"
                if fail_fast:
                    raise click.ClickException(msg)
                else:
                    console.print(msg, style="bold red")
            else:
                if not silent:
                    console.print(result.stdout)
                    console.log(
                        f"Command succeeded in {package.relative_to(obj['repo_root'])}"
                    )
