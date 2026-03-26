from pathlib import Path

import click

from llama_dev.utils import (
    BumpType,
    bump_version,
    load_pyproject,
    update_pyproject_version,
)


def _replace_core_dependency(project_path: Path, old_dep: str, new_dep: str):
    pyproject_path = project_path / "pyproject.toml"
    # Read the file content
    with open(pyproject_path, "r") as f:
        content = f.read()

    # Replace the old dependency string
    new_content = content.replace(old_dep, new_dep)

    # Write the updated content back
    with open(pyproject_path, "w") as f:
        f.write(new_content)


@click.command(
    short_help="Bump the versions to begin a llama_index umbrella package release"
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
def prepare(
    obj: dict,
    version_type: str,
    dry_run: bool,
):
    """Bump the version numbers to initiate the llama_index umbrella package release."""
    console = obj["console"]
    repo_root = obj["repo_root"]
    bump_enum = BumpType(version_type)

    root_package_data = load_pyproject(repo_root)
    current_version = root_package_data["project"]["version"]
    new_version = bump_version(current_version, bump_enum)
    new_dep_string = (
        f"llama-index-core>={new_version},<{bump_version(new_version, BumpType.MINOR)}"
    )

    if dry_run:
        console.print(f"Would bump llama_index from {current_version} to {new_version}")
        console.print(f"llama_index will depend on '{new_dep_string}'")
    else:
        # Update llama-index version number
        update_pyproject_version(repo_root, new_version)
        # Update llama-index-core version number
        update_pyproject_version(repo_root / "llama-index-core", new_version)
        # Update llama-index-core dependency version
        for dep in root_package_data["project"]["dependencies"]:
            if dep.startswith("llama-index-core"):
                _replace_core_dependency(repo_root, dep, new_dep_string)
                break
