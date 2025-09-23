import json
import re
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


def _get_version_from_init(repo_root: Path) -> str:
    init_py_path = (
        repo_root / "llama-index-core" / "llama_index" / "core" / "__init__.py"
    )

    init_py_content = init_py_path.read_text()
    match = re.search(r'__version__ = "(.+)"', init_py_content)
    if match is None:
        raise click.ClickException(f"Could not find __version__ in {init_py_path}")
    return match.group(1)


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
    - llama-index-core/pyproject.toml and llama-index-core/llama_index/core/__init__.py are consistent
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

    # Check consistency between pyproject.toml and __init__.py
    pyproject_version = _get_version_from_pyproject(repo_root)
    init_py_version = _get_version_from_init(repo_root)

    if pyproject_version != init_py_version:
        console.print(
            f"❌ Version mismatch between 'pyproject.toml' ({pyproject_version}) and "
            f"'__init__.py' ({init_py_version})",
            style="error",
        )
        exit(1)
    console.print(
        f"✅ Versions in 'pyproject.toml' and '__init__.py' are consistent ({pyproject_version})"
    )

    # Check if llama-index-core version is newer than PyPI
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
