#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import re
import subprocess as sp
import tomllib as toml
from pathlib import Path
from typing import Literal, NamedTuple


class Version(NamedTuple):
    major: int
    minor: int
    patch: int
    extra: str


def uv_lock(cwd: str) -> None:
    sp.run(["uv", "lock"], cwd=cwd, capture_output=True)


def load_version_from_toml(path: str) -> tuple[str, str]:
    with open(path, "rb") as f:
        data = toml.load(f)
    with open(path, "r") as f:
        content = f.read()
    assert isinstance(data, dict)
    assert "project" in data
    assert isinstance(data["project"], dict)
    assert "version" in data["project"]
    assert isinstance(data["project"]["version"], str)
    return data["project"]["version"], content


def parse_semver(version: str) -> Version:
    sep = version.split(".")
    assert len(sep) >= 3
    assert all(s.isdigit() for s in sep[:3])
    if len(sep) == 3:
        return Version(
            major=int(sep[0]), minor=int(sep[1]), patch=int(sep[2]), extra=""
        )
    return Version(
        major=int(sep[0]), minor=int(sep[1]), patch=int(sep[2]), extra=".".join(sep[3:])
    )


def semver_to_str(v: Version) -> str:
    if v.extra == "":
        return f"{v.major}.{v.minor}.{v.patch}"
    return f"{v.major}.{v.minor}.{v.patch}.{v.extra}"


def bump_semver(version: Version, bump_type: Literal["major", "minor", "patch"]) -> str:
    major = version.major
    minor = version.minor
    patch = version.patch
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return semver_to_str(Version(major=major, minor=minor, patch=patch, extra=""))


def new_version_to_toml(path: str, content: str, new_version: str) -> None:
    content = re.sub(
        r"^version\s*=\s*\"[^\"]+\"$",
        f'version = "{new_version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    with open(path, "w") as f:
        f.write(content)


def process_one_pyproject(path: str) -> None:
    version, content = load_version_from_toml(path)
    semver = parse_semver(version)
    bumped = bump_semver(semver, "minor")
    new_version_to_toml(path, content, bumped)
    parent = Path(path).parent
    uv_lock(str(parent))


def get_all_modified() -> list[str]:
    result = sp.run(["git", "status", "--short"], capture_output=True)
    modified = []
    base_path = Path.cwd()
    if result.returncode == 0:
        output = str(result.stdout, encoding="utf-8")
        lines = output.splitlines()
        for line in lines:
            if (
                "pyproject.toml" in line
                and "llama-index-core/" not in line
                and "llama-index-instrumentation/" not in line
            ):
                line = line.replace("M ", "").strip()
                path = base_path / line
                modified.append(str(path))
    return modified


def main() -> None:
    modified = get_all_modified()
    for i, m in enumerate(modified):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(modified)} packages...")
        process_one_pyproject(m)


if __name__ == "__main__":
    main()
