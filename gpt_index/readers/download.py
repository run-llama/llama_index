"""Download loader from the Loader Hub."""

import base64
import json
import os
import subprocess
import sys
from importlib import util
from pathlib import Path

import pkg_resources
import requests
from pkg_resources import DistributionNotFound

from gpt_index.readers.base import BaseReader

LOADER_HUB_URL = "https://api.github.com/repos/ahmetkca/llama-hub/contents/loader_hub{path}?ref=github-reader"


def _get_file_content(path: str) -> bool:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(LOADER_HUB_URL.format(path=path), headers=headers).json()
    print(resp)
    content = resp["content"]
    assert resp["encoding"] == "base64"
    return base64.b64decode(content).decode("utf-8")


def download_loader(loader_class: str) -> BaseReader:
    """Download a single loader from the Loader Hub.

    Args:
        loader_class: The name of the loader class you want to download,
            such as `SimpleWebPageReader`.
    Returns:
        A Loader.
    """
    library = json.loads(_get_file_content("/library.json"))

    # Look up the loader id (e.g. `web/simple_web`)
    loader_id = library[loader_class]["id"]
    extra_files = library[loader_class].get("extra_files", [])
    dirpath = "modules"
    loader_path = f"{dirpath}/{loader_id}"
    requirements_path = f"{loader_path}/requirements.txt"

    if not os.path.exists(dirpath):
        # Create a new directory because it does not exist
        os.makedirs(dirpath)

    # Create an __init__.py file if it does not exist under the modules directory
    with open(f"{dirpath}/__init__.py", "w") as f:
        f.write(f'""" Init file for modules directory. """')

    if not os.path.exists(loader_path):
        os.makedirs(loader_path)
        with open(f"{loader_path}/__init__.py", "w") as f:
            f.write(f'""" Init file for {loader_id} reader."""')
        with open(f"{loader_path}/base.py", "w") as f:
            f.write(_get_file_content(f"/{loader_id}/base.py"))
        for extra_file in extra_files:
            with open(f"{loader_path}/{extra_file}", "w") as f:
                f.write(_get_file_content(f"/{loader_id}/{extra_file}"))

    if not os.path.exists(requirements_path):
        with open(requirements_path, "w") as f:
            f.write(_get_file_content(f"/{loader_id}/requirements.txt"))

    # Install dependencies if there are any and not already installed
    if os.path.exists(requirements_path):
        try:
            requirements = pkg_resources.parse_requirements(
                Path(requirements_path).open()
            )
            pkg_resources.require([str(r) for r in requirements])
        except DistributionNotFound:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", requirements_path]
            )

    spec = util.spec_from_file_location(
        "custom_loader", location=f"{loader_path}/base.py"
    )
    if spec is None:
        raise ValueError(f"Could not find file: {loader_path}/base.py.")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    return getattr(module, loader_class)
