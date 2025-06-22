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
    package_has_tests,
)


class ResultStatus(Enum):
    """Represents the possible outcomes after shelling out pytest."""

    INSTALL_FAILED = auto()
    TESTS_FAILED = auto()
    TESTS_PASSED = auto()
    SKIPPED = auto()
    COVERAGE_FAILED = auto()


NO_TESTS_INDICATOR = "no tests ran"
MAX_CONSOLE_PRINT_LINES = 50


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
@click.option("--base-ref", required=False)
@click.option("--workers", default=8)
@click.argument("package_names", required=False, nargs=-1)
@click.pass_obj
def test(
    obj: dict,
    fail_fast: bool,
    cov: bool,
    cov_fail_under: int,
    base_ref: str,
    workers: int,
    package_names: tuple,
):
    # Fail on incompatible configurations
    if cov_fail_under and not cov:
        raise click.UsageError(
            "You have to pass --cov in order to use --cov-fail-under"
        )

    if base_ref is not None and not base_ref:
        raise click.UsageError("Option '--base-ref' cannot be empty.")

    if not package_names and not base_ref:
        raise click.UsageError(
            "Either pass '--base-ref' or provide at least one package name."
        )

    console = obj["console"]
    repo_root = obj["repo_root"]
    debug: bool = obj["debug"]
    packages_to_test: set[Path] = set()
    all_packages = find_all_packages(repo_root)

    if package_names:
        changed_packages: set[Path] = {repo_root / Path(pn) for pn in package_names}
    else:
        # Get the files that changed from the base branch
        changed_files = get_changed_files(repo_root, base_ref)
        # Get the packages containing the changed files
        changed_packages = get_changed_packages(changed_files, all_packages)

    # Find the dependants of the changed packages
    dependants = get_dependants_packages(changed_packages, all_packages)
    # Test the packages directly affected and their dependants
    packages_to_test = changed_packages | dependants

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
            package_name = package.relative_to(repo_root)
            if result["status"] == ResultStatus.INSTALL_FAILED:
                console.print(f"❗ Unable to build package {package_name}")
                console.print(
                    _trim(debug, f"Error:\n{result['stderr']}"), style="warning"
                )
            elif result["status"] == ResultStatus.TESTS_PASSED:
                console.print(f"✅ {package_name} succeeded in {result['time']}")
            elif result["status"] == ResultStatus.SKIPPED:
                console.print(f"⏭️  {package_name} skipped")
                console.print(
                    _trim(debug, f"Error:\n{result['stderr']}"), style="warning"
                )
            else:
                console.print(f"❌ {package_name} failed")
                console.print(
                    _trim(debug, f"Error:\n{result['stderr']}"), style="error"
                )
                console.print(
                    _trim(debug, f"Output:\n{result['stdout']}"), style="info"
                )

    # Print summary
    failed = [
        r["package"].relative_to(repo_root)
        for r in results
        if r["status"] in (ResultStatus.TESTS_FAILED, ResultStatus.COVERAGE_FAILED)
    ]
    install_failed = [
        r["package"].relative_to(repo_root)
        for r in results
        if r["status"] == ResultStatus.INSTALL_FAILED
    ]
    skipped = [
        r["package"].relative_to(repo_root)
        for r in results
        if r["status"] == ResultStatus.SKIPPED
    ]

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


def _trim(debug: bool, msg: str):
    lines = msg.split("\n")
    if len(lines) > MAX_CONSOLE_PRINT_LINES and not debug:
        lines = lines[:MAX_CONSOLE_PRINT_LINES]
        lines.append(
            "<-- llama-dev: output truncated, pass '--debug' to see the full log -->"
        )
    return "\n".join(lines)


def _uv_sync(
    package_path: Path, env: dict[str, str]
) -> subprocess.CompletedProcess:  # pragma: no cover
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
) -> subprocess.CompletedProcess:  # pragma: no cover
    """Run 'uv pip install -U <package_path1>, <package_path2>, ...' for locally changed packages."""
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
        "--no-sync",
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
) -> subprocess.CompletedProcess:  # pragma: no cover
    diff_cover_cmd = [
        "uv",
        "run",
        "--",
        "diff-cover",
        "coverage.xml",
        f"--fail-under={cov_fail_under}",
    ]
    if base_ref:
        diff_cover_cmd.append(f"--compare-branch={base_ref}")

    return subprocess.run(
        diff_cover_cmd,
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

    # Filter out packages that are not testable
    if not package_has_tests(package_path):
        return {
            "package": package_path,
            "status": ResultStatus.SKIPPED,
            "stdout": "",
            "stderr": f"Skipped: package has no tests",
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
    # Only fail if there are tests and they failed
    if result.returncode != 0 and NO_TESTS_INDICATOR not in str(result.stdout).lower():
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
