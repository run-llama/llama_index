import concurrent.futures
import os
import subprocess
import sys
import time
from enum import Enum, auto
from pathlib import Path

import click

from llama_dev.utils import (
    find_all_packages,
    get_changed_files,
    get_changed_packages,
    get_dep_names,
    get_dependants_packages,
    is_python_version_compatible,
    load_pyproject,
)


class ResultStatus(Enum):
    """Represents the possible outcomes after shelling out pytest."""

    INSTALL_FAILED = auto()
    TESTS_FAILED = auto()
    TESTS_PASSED = auto()
    SKIPPED = auto()
    COVERAGE_FAILED = auto()


@click.command(short_help="Run tests across the monorepo")
@click.option(
    "--fail-fast",
    is_flag=True,
    default=False,
    help="Exit the command at the first failure",
)
@click.option(
    "--cov",
    is_flag=True,
    default=False,
    help="Compute test coverage",
)
@click.option(
    "--cov-fail-under",
    default=0,
    help="Compute test coverage",
)
@click.option("--base-ref", required=True)
@click.option("--workers", default=8)
@click.pass_obj
def test(
    obj: dict,
    fail_fast: bool,
    cov: bool,
    cov_fail_under: int,
    base_ref: str,
    workers: int,
):
    # Fail on incompatible configurations
    if cov_fail_under and not cov:
        raise click.UsageError(
            "You have to pass --cov in order to use --cov-fail-under"
        )

    console = obj["console"]
    repo_root = obj["repo_root"]

    # Collect the packages to test
    all_packages = find_all_packages(repo_root)
    # Get the files that changed from the base branch
    changed_files = get_changed_files(repo_root, base_ref)
    # Get the packages containing the changed files
    changed_packages = get_changed_packages(changed_files, all_packages)
    # Find the dependants of the changed packages
    dependants = get_dependants_packages(changed_packages, all_packages)
    # Test the packages directly affected and their dependants
    packages_to_test: set[Path] = changed_packages | dependants

    # Test the packages using a process pool
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
        future_to_package = {
            executor.submit(
                _run_tests,
                package_path,
                changed_packages,
                base_ref,
                cov,
                cov_fail_under,
            ): package_path
            for package_path in sorted(packages_to_test)
        }

        for future in concurrent.futures.as_completed(future_to_package):
            result = future.result()
            results.append(result)

            # Print results as they complete
            package: Path = result["package"]
            if result["status"] == ResultStatus.INSTALL_FAILED:
                console.print(
                    f"❗ Unable to build package {package.relative_to(repo_root)}"
                )
                console.print(f"Error:\n{result['stderr']}", style="warning")
            elif result["status"] == ResultStatus.TESTS_PASSED:
                console.print(
                    f"✅ {package.relative_to(repo_root)} succeeded in {result['time']}"
                )
            elif result["status"] == ResultStatus.SKIPPED:
                console.print(f"⏭️  {package.relative_to(repo_root)} skipped")
                console.print(f"Error:\n{result['stderr']}", style="warning")
            else:
                console.print(f"❌ {package.relative_to(repo_root)} failed")
                console.print(f"Error:\n{result['stderr']}", style="error")
                console.print(f"Output:\n{result['stdout']}", style="info")

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
        console.print(
            f"\n{len(skipped)} packages were skipped due to Python version incompatibility:"
        )
        for p in skipped:
            print(p)

    if install_failed:
        # Do not fail the CI if there was an installation problem: many packages cannot
        # be installed in the runners, something we should either fix or ignore.
        console.print(f"\n{len(install_failed)} packages could not be installed:")
        for p in install_failed:
            print(p)

    if failed:
        console.print(f"\n{len(failed)} packages had test failures:")
        for p in failed:
            console.print(p)
        exit(1)
    else:
        console.print(
            f"\nTests passed for {len(results) - len(skipped)} packages.", style="green"
        )


def _uv_sync(package_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run 'uv sync' on a package folder."""
    return subprocess.run(
        ["uv", "sync"],
        cwd=package_path,
        text=True,
        capture_output=True,
        env=env,
    )


def _uv_install_local(
    package_path: Path, env: dict[str, str], install_local: set[Path]
) -> subprocess.CompletedProcess:
    """Run 'uv pip install -U <packge_path1>, <package_path2>, ...' for locally changed packages."""
    return subprocess.run(
        ["uv", "pip", "install", "-U", *install_local],
        cwd=package_path,
        text=True,
        capture_output=True,
        env=env,
    )


def _pytest(
    package_path: Path, env: dict[str, str], cov: bool
) -> subprocess.CompletedProcess:
    pytest_cmd = [
        "uv",
        "run",
        "--",
        "pytest",
        "-q",
        "--disable-warnings",
        "--disable-pytest-warnings",
    ]
    if cov:
        pytest_cmd += ["--cov=.", "--cov-report=xml"]

    return subprocess.run(
        pytest_cmd,
        cwd=package_path,
        text=True,
        capture_output=True,
        env=env,
    )


def _diff_cover(
    package_path: Path, env: dict[str, str], cov_fail_under: int, base_ref: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "uv",
            "run",
            "--",
            "diff-cover",
            "coverage.xml",
            f"--fail-under={cov_fail_under}",
            f"--compare-branch={base_ref}",
        ],
        cwd=package_path,
        text=True,
        capture_output=True,
        env=env,
    )


def _run_tests(
    package_path: Path,
    changed_packages: set[Path],
    base_ref: str,
    cov: bool,
    cov_fail_under: int,
):
    # Check Python version compatibility first
    package_data = load_pyproject(package_path)
    if not is_python_version_compatible(package_data):
        return {
            "package": package_path,
            "status": ResultStatus.SKIPPED,
            "stdout": "",
            "stderr": f"Skipped: Not compatible with Python {sys.version_info.major}.{sys.version_info.minor}",
            "time": "0.00s",
        }

    # Do not use the virtual environment calling llama-dev, if any
    env = os.environ.copy()
    if "VIRTUAL_ENV" in env:
        del env["VIRTUAL_ENV"]

    start = time.perf_counter()

    # Install dependencies
    result = _uv_sync(package_path, env)
    if result.returncode != 0:
        elapsed_time = time.perf_counter() - start
        return {
            "package": package_path,
            "status": ResultStatus.INSTALL_FAILED,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "time": f"{elapsed_time:.2f}s",
        }

    # Install local copies of packages that have changed
    dep_names = get_dep_names(package_data)
    install_local: set[Path] = set()
    for name in dep_names:
        for ppath in changed_packages:
            if ppath.name == name:
                install_local.add(ppath)
    if install_local:
        result = _uv_install_local(package_path, env, install_local)
        if result.returncode != 0:
            elapsed_time = time.perf_counter() - start
            return {
                "package": package_path,
                "status": ResultStatus.INSTALL_FAILED,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "time": f"{elapsed_time:.2f}s",
            }

    # Run pytest
    result = _pytest(package_path, env, cov)
    if result.returncode != 0:
        elapsed_time = time.perf_counter() - start
        return {
            "package": package_path,
            "status": ResultStatus.TESTS_FAILED,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "time": f"{elapsed_time:.2f}s",
        }

    # Check coverage report
    if cov_fail_under > 0:
        result = _diff_cover(package_path, env, cov_fail_under, base_ref)
        if result.returncode != 0:
            elapsed_time = time.perf_counter() - start
            return {
                "package": package_path,
                "status": ResultStatus.COVERAGE_FAILED,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "time": f"{elapsed_time:.2f}s",
            }

    # All done
    elapsed_time = time.perf_counter() - start
    return {
        "package": package_path,
        "status": ResultStatus.TESTS_PASSED,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "time": f"{elapsed_time:.2f}s",
    }
