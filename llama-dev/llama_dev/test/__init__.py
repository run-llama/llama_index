import concurrent.futures
import os
import subprocess
import sys
import time
from enum import Enum, auto
from pathlib import Path

import click
from rich.live import Live
from rich.table import Table

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
    NO_TESTS = auto()
    UNSUPPORTED_PYTHON_VERSION = auto()
    COVERAGE_FAILED = auto()


NO_TESTS_INDICATOR = "no tests ran"
MAX_CONSOLE_PRINT_LINES = 50


def _generate_status_table(
    total: int,
    completed: int,
    passed: int,
    failed: int,
    skipped: int,
    running_packages: list[str],
) -> Table:
    """Generate a Rich table showing test progress."""
    table = Table(title="Test Progress", show_header=True, header_style="bold cyan")
    table.add_column("Status", style="dim", width=20)
    table.add_column("Count", justify="right")

    table.add_row("Total Packages", str(total))
    table.add_row("Completed", f"{completed}/{total}")
    table.add_row("✅ Passed", str(passed), style="green")
    table.add_row("❌ Failed", str(failed), style="red")
    table.add_row("⏭️  Skipped", str(skipped), style="yellow")
    table.add_row("", "")

    if running_packages:
        table.add_row("Currently Running:", "", style="bold blue")
        for pkg in running_packages[:10]:  # Show max 10
            table.add_row("  →", pkg, style="blue")
        if len(running_packages) > 10:
            table.add_row(
                "  →", f"... and {len(running_packages) - 10} more", style="dim blue"
            )

    return table


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
    # Skip dependants if we're checking coverage
    if cov:
        dependants = set()
    else:
        dependants = get_dependants_packages(changed_packages, all_packages)

    # Test the packages directly affected and their dependants
    packages_to_test = changed_packages | dependants

    # Test the packages using a process pool
    results = []
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(workers))
    watchdog_triggered = False
    try:
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

        # Track last progress time for watchdog (detects if NO futures complete for too long)
        last_progress_time = time.time()
        watchdog_timeout = 600  # 10 minutes of no progress = stuck

        # Detect if we're in a CI environment (GitHub Actions, etc.)
        is_ci = (
            os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
        )

        if is_ci:
            # In CI: print periodic status updates instead of live display
            console.print(
                f"🚀 Starting tests for {len(packages_to_test)} packages with {workers} workers...\n"
            )
            last_update_time = time.time()
            update_interval = 30  # Print status every 30 seconds

            pending_futures = set(future_to_package.keys())

            while pending_futures:
                # Wait for futures with a timeout so we can print periodic updates
                done, pending_futures = concurrent.futures.wait(
                    pending_futures,
                    timeout=update_interval,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # Process completed futures
                for future in done:
                    try:
                        result = future.result(
                            timeout=0.1
                        )  # Should be immediate since future is done
                        results.append(result)
                        last_progress_time = time.time()  # Update progress time
                    except Exception as e:
                        # Handle any errors
                        pkg = future_to_package[future]
                        result = {
                            "package": pkg,
                            "status": ResultStatus.TESTS_FAILED,
                            "stdout": "",
                            "stderr": f"Test failed with exception: {e}",
                            "time": "N/A",
                        }
                        results.append(result)
                        failed_count += 1
                        last_progress_time = time.time()  # Update progress time
                        continue

                    # Update counts
                    if result["status"] == ResultStatus.TESTS_PASSED:
                        passed_count += 1
                    elif result["status"] in (
                        ResultStatus.TESTS_FAILED,
                        ResultStatus.COVERAGE_FAILED,
                        ResultStatus.INSTALL_FAILED,
                    ):
                        failed_count += 1
                    else:
                        skipped_count += 1

                # Check for stuck system (watchdog) - no progress for too long
                current_time = time.time()
                time_since_progress = current_time - last_progress_time

                if time_since_progress > watchdog_timeout and pending_futures:
                    console.print(
                        f"\n⚠️  Watchdog: No progress for {time_since_progress:.0f}s (>{watchdog_timeout}s), marking remaining tests as failed",
                        style="bold red",
                    )
                    # Mark all pending futures as failed
                    for f in list(pending_futures):
                        pkg = future_to_package[f]
                        console.print(f"   → {pkg.relative_to(repo_root)}", style="red")
                        # Mark as failed
                        result = {
                            "package": pkg,
                            "status": ResultStatus.TESTS_FAILED,
                            "stdout": "",
                            "stderr": f"Test watchdog timeout - no progress for {time_since_progress:.0f}s",
                            "time": "N/A",
                        }
                        results.append(result)
                        failed_count += 1
                    # Clear pending futures to exit loop
                    pending_futures.clear()
                    watchdog_triggered = True
                    break

                # Print status update
                running_packages = [
                    str(future_to_package[f].relative_to(repo_root))
                    for f in pending_futures
                ]

                console.print(
                    f"\n📊 Progress: {len(results)}/{len(packages_to_test)} | ✅ {passed_count} | ❌ {failed_count} | ⏭️ {skipped_count}"
                )
                if running_packages:
                    console.print(f"🔄 Currently running ({len(running_packages)}):")
                    for pkg in running_packages[:15]:  # Show up to 15 in CI
                        console.print(f"   → {pkg}")
                    if len(running_packages) > 15:
                        console.print(f"   ... and {len(running_packages) - 15} more")
                console.print()
                last_update_time = current_time
        else:
            # Local: Use Rich Live display to show progress
            update_interval = 2  # Update display every 2 seconds

            pending_futures = set(future_to_package.keys())

            with Live(
                _generate_status_table(
                    len(packages_to_test),
                    0,
                    0,
                    0,
                    0,
                    [
                        str(p.relative_to(repo_root))
                        for p in sorted(packages_to_test)[: int(workers)]
                    ],
                ),
                console=console,
                refresh_per_second=2,
            ) as live:
                while pending_futures:
                    # Wait for futures with a short timeout for responsive display
                    done, pending_futures = concurrent.futures.wait(
                        pending_futures,
                        timeout=update_interval,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    # Process completed futures
                    for future in done:
                        try:
                            result = future.result(timeout=0.1)  # Should be immediate
                            results.append(result)
                            last_progress_time = time.time()  # Update progress time
                        except Exception as e:
                            # Handle any errors
                            pkg = future_to_package[future]
                            result = {
                                "package": pkg,
                                "status": ResultStatus.TESTS_FAILED,
                                "stdout": "",
                                "stderr": f"Test failed with exception: {e}",
                                "time": "N/A",
                            }
                            results.append(result)
                            failed_count += 1
                            last_progress_time = time.time()  # Update progress time
                            continue

                        # Update counts
                        if result["status"] == ResultStatus.TESTS_PASSED:
                            passed_count += 1
                        elif result["status"] in (
                            ResultStatus.TESTS_FAILED,
                            ResultStatus.COVERAGE_FAILED,
                            ResultStatus.INSTALL_FAILED,
                        ):
                            failed_count += 1
                        else:
                            skipped_count += 1

                    # Check for stuck system (watchdog) - no progress for too long
                    current_time = time.time()
                    time_since_progress = current_time - last_progress_time

                    if time_since_progress > watchdog_timeout and pending_futures:
                        # Mark all pending futures as failed
                        for f in list(pending_futures):
                            pkg = future_to_package[f]
                            # Mark as failed
                            result = {
                                "package": pkg,
                                "status": ResultStatus.TESTS_FAILED,
                                "stdout": "",
                                "stderr": f"Test watchdog timeout - no progress for {time_since_progress:.0f}s",
                                "time": "N/A",
                            }
                            results.append(result)
                            failed_count += 1
                        # Clear pending futures to exit loop
                        pending_futures.clear()
                        watchdog_triggered = True
                        break

                    # Get currently running packages and update display
                    running_packages = [
                        str(future_to_package[f].relative_to(repo_root))
                        for f in pending_futures
                    ]

                    live.update(
                        _generate_status_table(
                            len(packages_to_test),
                            len(results),
                            passed_count,
                            failed_count,
                            skipped_count,
                            running_packages,
                        )
                    )

        # Print detailed results after completion
        console.print("\n" + "=" * 60 + "\n")
        console.print("Detailed Results:", style="bold")
        console.print()

        for result in results:
            package: Path = result["package"]
            package_name = package.relative_to(repo_root)
            if result["status"] == ResultStatus.INSTALL_FAILED:
                console.print(f"❗ Unable to build package {package_name}")
                console.print(
                    _trim(debug, f"Error:\n{result['stderr']}"), style="warning"
                )
            elif result["status"] == ResultStatus.TESTS_PASSED:
                if debug:  # Only print passed tests in debug mode
                    console.print(f"✅ {package_name} succeeded in {result['time']}")
            elif result["status"] == ResultStatus.UNSUPPORTED_PYTHON_VERSION:
                if debug:
                    console.print(
                        f"⏭️ {package_name} skipped due to python version incompatibility"
                    )
                    console.print(
                        _trim(debug, f"Error:\n{result['stderr']}"), style="warning"
                    )
            elif result["status"] == ResultStatus.NO_TESTS:
                if debug:
                    console.print(f"⏭️ {package_name} skipped due to no tests")
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
    finally:
        # Shutdown executor
        if watchdog_triggered:
            # Don't wait for hung futures
            console.print(
                "\n⚠️  Skipping executor shutdown (watchdog triggered)", style="dim"
            )
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            # Normal shutdown
            executor.shutdown(wait=True)

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
    skipped_no_tests = [
        r["package"].relative_to(repo_root)
        for r in results
        if r["status"] == ResultStatus.NO_TESTS
        and "package has no tests" in r["stderr"]
    ]
    skipped_pyversion_incompatible = [
        r["package"].relative_to(repo_root)
        for r in results
        if r["status"] == ResultStatus.UNSUPPORTED_PYTHON_VERSION
        and "Not compatible with Python" in r["stderr"]
    ]

    if skipped_pyversion_incompatible:
        console.print(
            f"\n{len(skipped_pyversion_incompatible)} packages were skipped due to Python version incompatibility:"
        )
        for p in skipped_pyversion_incompatible:
            print(p)
    if skipped_no_tests:
        console.print(
            f"\n{len(skipped_no_tests)} packages were skipped because they have no tests:"
        )
        for p in skipped_no_tests:
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
        # Use os._exit() if watchdog triggered to avoid hanging on cleanup
        if watchdog_triggered:
            console.print(
                "\n⚠️  Using immediate exit due to watchdog timeout", style="dim"
            )
            os._exit(1)
        else:
            exit(1)
    else:
        console.print(
            f"\nTests passed for {len(results) - len(skipped_no_tests) - len(skipped_pyversion_incompatible)} packages.",
            style="green",
        )
        # Use os._exit() if watchdog triggered to avoid hanging on cleanup
        if watchdog_triggered:
            console.print(
                "\n⚠️  Using immediate exit due to watchdog timeout", style="dim"
            )
            os._exit(0)


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
            "status": ResultStatus.UNSUPPORTED_PYTHON_VERSION,
            "stdout": "",
            "stderr": f"Skipped: Not compatible with Python {sys.version_info.major}.{sys.version_info.minor}",
            "time": "0.00s",
        }

    # Filter out packages that are not testable
    if not package_has_tests(package_path):
        return {
            "package": package_path,
            "status": ResultStatus.NO_TESTS,
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
