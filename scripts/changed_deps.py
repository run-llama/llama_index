import argparse
import concurrent.futures
import os
import subprocess
import time
from enum import Enum, auto
from pathlib import Path
from typing import List, Set


class ResultStatus(Enum):
    INSTALL_FAILED = auto()
    TESTS_FAILED = auto()
    TESTS_PASSED = auto()


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
    return parser.parse_args()


def get_changed_files(base_ref: str, repo_root: str) -> List[str]:
    """Get list of files changed compared to the base branch."""
    try:
        cmd = ["git", "diff", "--name-only", f"{base_ref}...HEAD"]
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")

        return [f for f in result.stdout.splitlines() if f.strip()]
    except Exception as e:
        print(f"Exception occurred: {e!s}")
        raise


def find_integrations(root_path: str, recursive=False) -> list[Path]:
    """Find all integrations packages in the repo."""
    package_roots: list[Path] = []
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
                package_roots.append(package_name)

    return package_roots


def find_packs(root_path: str) -> list[Path]:
    """Find all llama-index-packs packages in the repo."""
    package_roots = []
    packs_root = Path(root_path) / "llama-index-packs"

    for package_name in packs_root.iterdir():
        tests_folder = package_name / "tests"
        if package_name.is_dir() and tests_folder.exists():
            package_roots.append(package_name)

    return package_roots


def find_utils(root_path: str) -> list[Path]:
    """Find all llama-index-utils packages in the repo."""
    package_roots = []
    utils_root = Path(root_path) / "llama-index-utils"

    for package_name in utils_root.iterdir():
        tests_folder = package_name / "tests"
        if package_name.is_dir() and tests_folder.exists():
            package_roots.append(package_name)

    return package_roots


def get_affected_packages(
    changed_files: list[str], all_packages: list[Path]
) -> Set[str]:
    """Get packages containing the files in the changeset."""
    affected_packages = set()

    for file_path in changed_files:
        # Find the package containing this file
        for pkg_dir in all_packages:
            if file_path.startswith(str(pkg_dir)):
                affected_packages.add(pkg_dir)
                break

    return affected_packages


def run_pytest(package_dir):
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

    result = subprocess.run(
        [
            "uv",
            "run",
            "--",
            "pytest",
            "-q",
            "--disable-warnings",
            "--disable-pytest-warnings",
        ],
        cwd=package_dir,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode == 0:
        status = ResultStatus.TESTS_PASSED
    else:
        status = ResultStatus.TESTS_FAILED

    elapsed_time = time.perf_counter() - start
    return {
        "package": str(package_dir),
        "status": status,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "time": f"{elapsed_time:.2f}s",
    }


def main():
    args = parse_args()

    # Get changed files
    changed_files = get_changed_files(args.base_ref, args.repo_root)

    # Find all packages in the repo
    all_packages = [
        Path("llama-index-core"),
        *find_integrations(args.repo_root),
        Path("llama-index-networks"),
        *find_packs(args.repo_root),
        *find_utils(args.repo_root),
    ]

    # Find directly affected packages
    directly_affected = get_affected_packages(changed_files, all_packages)
    if "llama-index-core" in directly_affected:
        # Run all the tests if llama-index-core was changed
        packages_to_test = {str(p) for p in all_packages}
    else:
        packages_to_test = directly_affected

    # Run pytest for each affected package in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        future_to_package = {
            executor.submit(run_pytest, package): package
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
            else:
                print(f"❌ {package} failed")
                print(f"Error:\n{result['stderr']}")
                print(f"Output:\n{result['stdout']}")

    # Print summary
    failed = [r["package"] for r in results if r["status"] == ResultStatus.TESTS_FAILED]
    install_failed = [
        r["package"] for r in results if r["status"] == ResultStatus.INSTALL_FAILED
    ]
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
        print(f"\nTests passed for {len(results)} packages.")


if __name__ == "__main__":
    main()
