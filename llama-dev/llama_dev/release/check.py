import json
import subprocess
import urllib.request
from pathlib import Path

import click
import tomli
from packaging.version import parse as parse_version


def _get_current_branch_name() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )


def _get_version_from_pyproject(repo_root: Path) -> str:
    with open(repo_root / "llama-index-core" / "pyproject.toml", "rb") as f:
        pyproject_data = tomli.load(f)

    return pyproject_data["project"]["version"]


def _get_version_from_pypi() -> str:
    try:
        url = "https://pypi.org/pypi/llama-index-core/json"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.load(response)
        return data["info"]["version"]
    except Exception as e:
        raise click.ClickException(
            f"Failed to fetch llama-index-core version from PyPI: {e}"
        )


@click.command(short_help="Check if requisites for the release are satisfied")
@click.option(
    "--before-core",
    is_flag=True,
    help="Run the check during pre-release (before releasing llama-index-core)",
    default=False,
)
@click.pass_obj
def check(obj: dict, before_core: bool):
    """
    Check if all the requisites for the release are satisfied.

    \b
    Requisites before releasing llama-index-core (passing --before-core):
    - llama-index-core/pyproject.toml is newer than the latest on PyPI

    Requisite after llama-index-core was published (without passing --before-core):
    - current branch is `main`
    - version from llama-index-core/pyproject.toml is the latest on PyPI
    """  # noqa
    console = obj["console"]
    repo_root = obj["repo_root"]

    current_branch = _get_current_branch_name()
    # Check current branch IS main
    if current_branch != "main":
        console.print(
            "❌ To release 'llama-index' you have to checkout the `main` branch.",
            style="error",
        )
        exit(1)
    console.print("✅ You are on the `main` branch.")

    if before_core:
        # Check llama-index-core version is NEWER than PyPI
        pyproject_version = _get_version_from_pyproject(repo_root)
        pypi_version = _get_version_from_pypi()
        if not parse_version(pyproject_version) > parse_version(pypi_version):
            console.print(
                f"❌ Version {pyproject_version} is not newer than the latest on PyPI ({pypi_version}).",
                style="error",
            )
            exit(1)
        console.print(
            f"✅ Version {pyproject_version} is newer than the latest on PyPI ({pypi_version})."
        )
    else:
        # Check llama-index-core version is SAME as PyPI
        pyproject_version = _get_version_from_pyproject(repo_root)
        pypi_version = _get_version_from_pypi()
        if parse_version(pyproject_version) > parse_version(pypi_version):
            console.print(
                f"❌ Version {pyproject_version} is not available on PyPI.",
                style="error",
            )
            exit(1)
        console.print(f"✅ Version {pyproject_version} is the latest on PyPI.")
