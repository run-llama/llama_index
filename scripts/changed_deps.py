import argparse
import subprocess
from pathlib import Path
from typing import List, Set

IGNORE_FOR_TESTING = [
    ".",
    "docs",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find packages affected by changes since base branch"
    )
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Base branch to compare against (e.g., 'main')",
    )
    parser.add_argument(
        "--repo-root", default="./", help="Root directory of the repository"
    )
    return parser.parse_args()


def get_changed_files(base_ref: str, repo_root: str) -> List[str]:
    """Get list of files changed since the base branch."""
    try:
        cmd = ["git", "diff", "--name-only", f"{base_ref}...HEAD"]
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")

        return [f for f in result.stdout.splitlines() if f.strip()]
    except Exception as e:
        print(f"Exception occurred: {e!s}")
        raise


def find_package_roots(repo_root: str) -> list[str]:
    """
    Find all Python packages in the repo.
    Returns a dict mapping package names to their root directories.
    """
    roots = []

    # Look for directories with pyproject.toml files
    for pyproject_path in Path(repo_root).glob("**/pyproject.toml"):
        package_dir = str(pyproject_path.parent.relative_to(repo_root))
        if ".venv" in package_dir or package_dir in IGNORE_FOR_TESTING:
            continue
        roots.append(package_dir)

    return roots


def get_affected_packages(
    changed_files: list[str], all_packages: list[str]
) -> Set[str]:
    """Get packages containing changed files."""
    affected_packages = set()

    for file_path in changed_files:
        # Find the package containing this file
        for pkg_dir in all_packages:
            if file_path.startswith(pkg_dir):
                affected_packages.add(pkg_dir)
                break

    return affected_packages


def main():
    args = parse_args()

    # Get changed files
    changed_files = get_changed_files(args.base_ref, args.repo_root)

    # Find all packages in the repo
    all_packages = find_package_roots(args.repo_root)

    # Map files to directly affected packages
    directly_affected = get_affected_packages(changed_files, all_packages)

    if "llama-index-core" in directly_affected:
        directly_affected = all_packages

    print("\n".join(sorted(directly_affected)))


if __name__ == "__main__":
    main()
