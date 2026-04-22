#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Dependency hygiene helper for uv-managed monorepo packages.

This script is intentionally lightweight so contributors can run the same checks
locally and in CI:
- verify `uv.lock` matches `pyproject.toml` (`uv lock --check`)
- optionally refresh locks (`uv lock`)
- report directories that contain both `pyproject.toml` and requirements files
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def _project_dirs() -> list[Path]:
    dirs: list[Path] = []
    for pyproject in REPO_ROOT.rglob("pyproject.toml"):
        if ".venv" in pyproject.parts:
            continue
        dirs.append(pyproject.parent)
    return sorted(set(dirs))


def _changed_project_dirs(base_ref: str) -> list[Path]:
    diff = _run(["git", "diff", "--name-only", f"{base_ref}...HEAD"], cwd=REPO_ROOT)
    if diff.returncode != 0:
        raise RuntimeError(diff.stderr.strip() or "Unable to compute changed files")

    changed: set[Path] = set()
    for rel_path in [line.strip() for line in diff.stdout.splitlines() if line.strip()]:
        path = REPO_ROOT / rel_path
        if path.name in {"pyproject.toml", "uv.lock"} and path.exists():
            changed.add(path.parent)
    return sorted(changed)


def _scan_requirements_with_pyproject(target_dirs: list[Path]) -> list[Path]:
    offenders: list[Path] = []
    for directory in target_dirs:
        if not (directory / "pyproject.toml").exists():
            continue
        requirements_files = sorted(directory.glob("requirements*.txt"))
        if requirements_files:
            offenders.append(directory)
    return offenders


def _lock_action(target_dirs: list[Path], check_only: bool) -> int:
    failures: list[Path] = []
    command = ["uv", "lock", "--check"] if check_only else ["uv", "lock"]

    for directory in target_dirs:
        if not (directory / "uv.lock").exists():
            continue
        result = _run(command, cwd=directory)
        if result.returncode == 0:
            print(f"[ok] {' '.join(command)} :: {directory.relative_to(REPO_ROOT)}")
            continue
        failures.append(directory)
        print(f"[fail] {' '.join(command)} :: {directory.relative_to(REPO_ROOT)}")
        if result.stderr.strip():
            print(result.stderr.strip())

    if failures:
        print("\nLock action failed in:")
        for directory in failures:
            print(f"- {directory.relative_to(REPO_ROOT)}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run monorepo dependency hygiene checks.")
    parser.add_argument(
        "action",
        choices=["lock-check", "lock-refresh", "requirements-scan"],
        help="Which hygiene action to run.",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Scope actions to directories changed since --base-ref.",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git base ref used with --changed-only (default: origin/main).",
    )
    args = parser.parse_args()

    target_dirs = (
        _changed_project_dirs(args.base_ref) if args.changed_only else _project_dirs()
    )
    if not target_dirs:
        print("No target package directories found.")
        return 0

    if args.action == "requirements-scan":
        offenders = _scan_requirements_with_pyproject(target_dirs)
        if not offenders:
            print("No package directories found with both pyproject.toml and requirements*.txt.")
            return 0
        print("Directories containing both pyproject.toml and requirements*.txt:")
        for directory in offenders:
            print(f"- {directory.relative_to(REPO_ROOT)}")
        return 0

    if args.action == "lock-check":
        return _lock_action(target_dirs, check_only=True)
    if args.action == "lock-refresh":
        return _lock_action(target_dirs, check_only=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
