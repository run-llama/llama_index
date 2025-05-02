import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import tomli
from packaging import specifiers, version

DEP_NAME_REGEX = re.compile(r"([^<>=\s]+)")


class ResultStatus(Enum):
    INSTALL_FAILED = auto()
    TESTS_FAILED = auto()
    TESTS_PASSED = auto()
    SKIPPED = auto()
    COVERAGE_FAILED = auto()


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
    parser.add_argument(
        "--workers", default=8, help="Number of concurrent processes running pytest"
    )
    parser.add_argument("--cov-fail-under", help="Check coverage of changed files")
    return parser.parse_args()


def get_changed_files(base_ref: str, repo_root: str) -> list[str]:
    """Use git to get the list of files changed compared to the base branch."""
    try:
        cmd = ["git", "diff", "--name-only", f"{base_ref}...HEAD"]
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")

        return [f for f in result.stdout.splitlines() if f.strip()]
    except Exception as e:
        print(f"Exception occurred: {e!s}")
        raise


def get_changed_packages(changed_files: list[str], all_packages: list[str]) -> set[str]:
    """Get the packages containing the files in changed_files."""
    changed_packages = set()

    for file_path in changed_files:
        # Find the package containing this file
        for pkg_dir in all_packages:
            if file_path.startswith(str(pkg_dir)):
                changed_packages.add(str(pkg_dir))
                break

    return changed_packages


def load_pyproject(pyproject_path: Path) -> dict:
    """Thin wrapper around tomli.load()."""
    with open(pyproject_path, "rb") as f:
        return tomli.load(f)


def get_dep_names(pyproject_data: dict) -> set[str]:
    """Load dependencies from pyproject.toml."""
    dependencies: set[str] = set()
    for dep in pyproject_data["project"]["dependencies"]:
        matches = DEP_NAME_REGEX.findall(dep)
        if not matches:
            continue
        dependencies.add(matches[0])
    return dependencies


def is_python_version_compatible(pyproject_data: dict) -> bool:
    """Check if the package is compatible with the current Python version using packaging."""
    # Get current Python version
    current_version = version.Version(
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # Get Python version requirements if they exist
    requires_python = pyproject_data.get("project", {}).get("requires-python", None)

    if not requires_python:
        # If no Python version is specified, assume it's compatible
        return True

    try:
        # Parse the version specifier
        spec = specifiers.SpecifierSet(requires_python)

        # Check if the current version satisfies the specifier
        return spec.contains(str(current_version))
    except Exception as e:
        # If there's any error in parsing, log it and assume compatibility
        print(
            f"Warning: Could not parse Python version specifier '{requires_python}': {e}"
        )
        return True


def get_dependants_packages(
    changed_packages: set[str], all_packages: list[str]
) -> set[str]:
    """Get packages containing the files in the changeset."""
    dependants_packages: set[str] = set()
    dep_name_re = re.compile(r"([^<>=\s]+)")

    changed_packages_names: set[str] = set()
    for pkg_path in changed_packages:
        pyproject_data = load_pyproject(Path(pkg_path) / "pyproject.toml")
        changed_packages_names.add(pyproject_data["project"]["name"])

    for pkg_path in all_packages:
        pyproject_data = load_pyproject(Path(pkg_path) / "pyproject.toml")
        for dep_name in get_dep_names(pyproject_data):
            if dep_name in changed_packages_names:
                dependants_packages.add(str(pkg_path))

    return dependants_packages


def find_integrations(root_path: str, recursive=False) -> list[str]:
    """Find all integrations packages in the repo."""
    package_roots: list[str] = []
    integrations_root = Path(root_path)
    if not recursive:
        integrations_root = integrations_root / "llama-index-integrations"

    for category_path in integrations_root.iterdir():
        if not category_path.is_dir():
            continue

        if category_path.name == "storage":
            # The "storage" category has sub-folders
            package_roots += find_integrations(str(category_path), recursive=True)
            continue

        for package_name in category_path.iterdir():
            tests_folder = package_name / "tests"
            if package_name.is_dir() and tests_folder.exists():
                package_roots.append(str(package_name))

    return package_roots


def find_packs(root_path: str) -> list[str]:
    """Find all llama-index-packs packages in the repo."""
    package_roots: list[str] = []
    packs_root = Path(root_path) / "llama-index-packs"

    for package_name in packs_root.iterdir():
        tests_folder = package_name / "tests"
        if package_name.is_dir() and tests_folder.exists():
            package_roots.append(str(package_name))

    return package_roots


def find_utils(root_path: str) -> list[str]:
    """Find all llama-index-utils packages in the repo."""
    package_roots: list[str] = []
    utils_root = Path(root_path) / "llama-index-utils"

    for package_name in utils_root.iterdir():
        tests_folder = package_name / "tests"
        if package_name.is_dir() and tests_folder.exists():
            package_roots.append(str(package_name))

    return package_roots


def run_pytest(
    root_dir: str,
    package_dir: str,
    changed_packages: set[str],
    base_ref: str,
    cov_fail_under: Optional[int] = None,
):
    # Check Python version compatibility first
    pyproject_path = Path(package_dir) / "pyproject.toml"
    if pyproject_path.exists():
        pyproject_data = load_pyproject(pyproject_path)
        if not is_python_version_compatible(pyproject_data):
            return {
                "package": str(package_dir),
                "status": ResultStatus.SKIPPED,
                "stdout": "",
                "stderr": f"Skipped: Not compatible with Python {sys.version_info.major}.{sys.version_info.minor}",
                "time": "0.00s",
            }

    env = os.environ.copy()
    if "VIRTUAL_ENV" in env:
        del env["VIRTUAL_ENV"]

    start = time.perf_counter()

    result = subprocess.run(
        [
            "uv",
            "sync",
        ],
        cwd=package_dir,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        elapsed_time = time.perf_counter() - start
        return {
            "package": str(package_dir),
            "status": ResultStatus.INSTALL_FAILED,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "time": f"{elapsed_time:.2f}s",
        }

    # Install local copy of packages that have changed
    pyproject_data = load_pyproject(Path(package_dir) / "pyproject.toml")
    dep_names = get_dep_names(pyproject_data)
    install_local = set()
    for name in dep_names:
        for package_path in changed_packages:
            if package_path.endswith(name):
                install_local.add(str(Path(root_dir) / package_path))
    if install_local:
        result = subprocess.run(
            ["uv", "pip", "install", "-U", *install_local],
            cwd=package_dir,
            text=True,
            capture_output=True,
            env=env,
        )
        if result.returncode != 0:
            elapsed_time = time.perf_counter() - start
            return {
                "package": str(package_dir),
                "status": ResultStatus.INSTALL_FAILED,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "time": f"{elapsed_time:.2f}s",
            }

    pytest_cmd = [
        "uv",
        "run",
        "--",
        "pytest",
        "-q",
        "--disable-warnings",
        "--disable-pytest-warnings",
    ]
    if cov_fail_under is not None:
        pytest_cmd += [
            "--cov=.",
            "--cov-report=xml",
        ]
    result = subprocess.run(
        pytest_cmd,
        cwd=package_dir,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        elapsed_time = time.perf_counter() - start
        return {
            "package": str(package_dir),
            "status": ResultStatus.TESTS_FAILED,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "time": f"{elapsed_time:.2f}s",
        }

    if cov_fail_under is not None:
        result = subprocess.run(
            [
                "uv",
                "run",
                "--",
                "diff-cover",
                "coverage.xml",
                f"--fail-under={cov_fail_under}",
                f"--compare-branch={base_ref}",
            ],
            cwd=package_dir,
            text=True,
            capture_output=True,
            env=env,
        )
        if result.returncode != 0:
            elapsed_time = time.perf_counter() - start
            return {
                "package": str(package_dir),
                "status": ResultStatus.COVERAGE_FAILED,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "time": f"{elapsed_time:.2f}s",
            }

    elapsed_time = time.perf_counter() - start
    return {
        "package": str(package_dir),
        "status": ResultStatus.TESTS_PASSED,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "time": f"{elapsed_time:.2f}s",
    }


def main():
    args = parse_args()

    # Find all packages in the repo
    all_packages = [
        Path("llama-index-core"),
        *find_integrations(args.repo_root),
        Path("llama-index-networks"),
        *find_packs(args.repo_root),
        *find_utils(args.repo_root),
    ]
    # Get the files that changed from the base branch
    changed_files = get_changed_files(args.base_ref, args.repo_root)
    # Get the packages containing the changed files
    changed_packages = get_changed_packages(changed_files, all_packages)
    # Find the dependants of the changed packages
    dependants = get_dependants_packages(changed_packages, all_packages)
    # Test the packages directly affected and their dependants
    packages_to_test: set[str] = changed_packages | dependants

    # Run pytest for each affected package in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        future_to_package = {
            executor.submit(
                run_pytest,
                str(Path(args.repo_root).resolve()),
                package,
                changed_packages,
                args.base_ref,
                args.cov_fail_under,
            ): package
            for package in sorted(packages_to_test)
        }

        for future in concurrent.futures.as_completed(future_to_package):
            result = future.result()
            results.append(result)

            # Print results as they complete
            package = result["package"]
            if result["status"] == ResultStatus.INSTALL_FAILED:
                print(f"❗ Unable to build package {package}")
                print(f"Error:\n{result['stderr']}")
            elif result["status"] == ResultStatus.TESTS_PASSED:
                print(f"✅ {package} succeeded in {result['time']}")
            elif result["status"] == ResultStatus.SKIPPED:
                print(f"⏭️  {package} skipped: {result['stderr']}")
            else:
                print(f"❌ {package} failed")
                print(f"Error:\n{result['stderr']}")
                print(f"Output:\n{result['stdout']}")

    # Print summary
    failed = [
        r["package"]
        for r in results
        if r["status"] in (ResultStatus.TESTS_FAILED, ResultStatus.COVERAGE_FAILED)
    ]
    install_failed = [
        r["package"] for r in results if r["status"] == ResultStatus.INSTALL_FAILED
    ]
    skipped = [r["package"] for r in results if r["status"] == ResultStatus.SKIPPED]

    if skipped:
        print(
            f"\n{len(skipped)} packages were skipped due to Python version incompatibility:"
        )
        for p in skipped:
            print(p)

    if install_failed:
        # Do not fail the CI if there was an installation problem: many packages cannot
        # be installed in the runners, something we should either fix or ignore.
        print(f"\n{len(install_failed)} packages could not be installed:")
        for p in install_failed:
            print(p)
    if failed:
        print(f"\n{len(failed)} packages had test failures:")
        for p in failed:
            print(p)
        exit(1)
    else:
        print(f"\nTests passed for {len(results) - len(skipped)} packages.")


if __name__ == "__main__":
    main()
