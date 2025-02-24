import tomli
import os
from pathlib import Path
from typing import List, Dict


def bump_minor_version(version: str) -> str:
    """Bumps the minor version of a version string (x.y.z -> x.(y+1).0)."""
    if len(version.split(".")) == 2:
        major, minor = version.split(".")
        return f"{major}.{int(minor) + 1}.0"

    major, minor, patch = version.split(".")
    return f"{major}.{int(minor) + 1}.0"


def update_package_versions(pyproject_paths: List[str]) -> None:
    # First pass: read all current versions and calculate new ones
    package_versions: Dict[
        str, tuple[str, str]
    ] = {}  # name -> (old_version, new_version)

    for path in pyproject_paths:
        with open(path, "rb") as f:
            data = tomli.load(f)

        package_name = data["tool"]["poetry"]["name"]
        current_version = data["tool"]["poetry"]["version"]
        new_version = bump_minor_version(current_version)
        package_versions[package_name] = (current_version, new_version)

    # Second pass: update files using string replacement to preserve formatting
    for path in pyproject_paths:
        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Load data just to get the package name
        with open(path, "rb") as f:
            data = tomli.load(f)
            package_name = data["tool"]["poetry"]["name"]

        # Update package's own version
        old_ver, new_ver = package_versions[package_name]
        content = content.replace(f'version = "{old_ver}"', f'version = "{new_ver}"')

        # Update llama-index-core version if present
        import re

        content = re.sub(
            r'llama-index-core = "\^0\.11\.\d+"',
            'llama-index-core = "^0.12.0"',
            content,
        )
        content = re.sub(
            r'llama-index-core = "\^0\.11"', 'llama-index-core = "^0.12.0"', content
        )

        # Update dependencies versions
        for dep, (old_ver, new_ver) in package_versions.items():
            if dep in content:
                content = re.sub(f'{dep} = "[^"]+"', f'{dep} = "^{new_ver}"', content)

        # Write updated content back to file
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Updated {path}")
        print(f"  New version: {new_ver}")


def main():
    # Example usage
    # Find all pyproject.toml files in specified directories
    base_dirs = [
        "llama-index-integrations",
        "llama-index-packs",
        "llama-datasets",
        "llama-index-cli",
        "llama-index-experimental",
        "llama-index-finetuning",
        "llama-index-networks",
        "llama-index-utils",
    ]

    pyproject_paths = ["./pyproject.toml"]
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            # Use rglob to recursively find all pyproject.toml files
            pyproject_paths.extend(
                str(p) for p in Path(base_dir).rglob("pyproject.toml")
            )

    # breakpoint()
    print(f"Found {len(pyproject_paths)} pyproject.toml files")
    update_package_versions(pyproject_paths)


if __name__ == "__main__":
    main()
