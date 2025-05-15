import re
from enum import Enum
from pathlib import Path

import click
from packaging.version import Version

from llama_dev.utils import find_all_packages, is_llama_index_package, load_pyproject


class BumpType(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


def bump_version(current_version: str, bump_type: BumpType) -> str:
    """Bump a version string according to semver rules."""
    v = Version(current_version)

    # Parse the version components
    release = v.release
    major = release[0] if len(release) > 0 else 0
    minor = release[1] if len(release) > 1 else 0
    micro = release[2] if len(release) > 2 else 0

    version_str = ""
    if bump_type == BumpType.MAJOR:
        version_str = f"{major + 1}.0.0"
    elif bump_type == BumpType.MINOR:
        version_str = f"{major}.{minor + 1}.0"
    elif bump_type == BumpType.PATCH:
        version_str = f"{major}.{minor}.{micro + 1}"

    return version_str


def update_pyproject_version(package_path: Path, new_version: str) -> None:
    """Update the version in a pyproject.toml file."""
    pyproject_path = package_path / "pyproject.toml"

    # Read the file content
    with open(pyproject_path, "r") as f:
        content = f.read()

    pattern = r'(\[project\][^\[]*?version\s*=\s*["\'])([^"\']+)(["\'])'
    new_content = re.sub(pattern, rf"\g<1>{new_version}\g<3>", content, flags=re.DOTALL)

    # Write the updated content back
    with open(pyproject_path, "w") as f:
        f.write(new_content)


@click.command(short_help="Bump package version")
@click.argument("package_names", required=False, nargs=-1)
@click.option(
    "--all",
    is_flag=True,
    help="Bump version for all the packages in the monorepo",
)
@click.option(
    "--version-type",
    type=click.Choice([t.value for t in BumpType], case_sensitive=False),
    default=BumpType.PATCH.value,
    help="Type of version bump to perform (default: patch)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes",
)
@click.pass_obj
def bump(
    obj: dict,
    all: bool,
    package_names: tuple,
    version_type: str,
    dry_run: bool,
):
    """Bump version for specified packages or all packages."""
    console = obj["console"]

    if not all and not package_names:
        raise click.UsageError("Either specify package name(s) or use the --all flag")

    packages = set()
    if all:
        packages = find_all_packages(obj["repo_root"])
    else:
        for package_name in package_names:
            package_path = obj["repo_root"] / package_name
            if not is_llama_index_package(package_path):
                raise click.UsageError(
                    f"{package_name} is not a path to a LlamaIndex package"
                )
            packages.add(package_path)

    bump_enum = BumpType(version_type)

    # First, collect all packages and their version changes
    changes = []
    for package in packages:
        try:
            package_data = load_pyproject(package)
            current_version = package_data["project"]["version"]
            new_version = bump_version(current_version, bump_enum)
            if dry_run:
                console.print(
                    f"Would bump {package.relative_to(obj['repo_root'])} from {current_version} to {new_version}"
                )
            else:
                update_pyproject_version(package, new_version)
        except Exception as e:
            console.print(
                f"[error]Error processing {package.relative_to(obj['repo_root'])}: {e!s}[/error]"
            )
