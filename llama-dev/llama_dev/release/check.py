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


@click.command(
    short_help="Check if all the pre-requisites for the release are satisfied"
)
@click.pass_obj
def check(obj: dict):
    """
    Check if all the pre-requisites for the release are satisfied.

    Pre-requisites:
    - current branch is not `main`
    - llama-index-core/pyproject.toml is newer than the latest on PyPI

    """
    console = obj["console"]
    repo_root = obj["repo_root"]

    # Check current branch is not main
    current_branch = _get_current_branch_name()
    if current_branch == "main":
        console.print(
            "❌ You are on the `main` branch. Please create a new branch to release.",
            style="error",
        )
        exit(1)
    console.print("✅ You are not on the `main` branch.")

    # Check if llama-index-core version is newer than PyPI
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
