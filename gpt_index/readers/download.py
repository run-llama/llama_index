"""Download loader from the Loader Hub."""

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

LOADER_HUB_URL = (
    "https://raw.githubusercontent.com/emptycrown/loader-hub/main/loader_hub"
)


def download_loader(loader_class: str) -> BaseReader:
    """Download a single loader from the Loader Hub.

    Args:
        loader_class: The name of the loader class you want to download,
            such as `SimpleWebPageReader`.
    Returns:
        A Loader.
    """
    response = requests.get(f"{LOADER_HUB_URL}/library.json")
    library = json.loads(response.text)

    # Look up the loader id (e.g. `web/simple_web`)
    loader_id = library[loader_class]["id"]
    dirpath = ".modules"
    loader_filename = loader_id.replace("/", "-")
    loader_path = f"{dirpath}/{loader_filename}.py"
    requirements_path = f"{dirpath}/{loader_filename}_requirements.txt"

    if not os.path.exists(dirpath):
        # Create a new directory because it does not exist
        os.makedirs(dirpath)

    if not os.path.exists(loader_path):
        response = requests.get(f"{LOADER_HUB_URL}/{loader_id}/base.py")
        with open(loader_path, "w") as f:
            f.write(response.text)

    if not os.path.exists(requirements_path):
        response = requests.get(f"{LOADER_HUB_URL}/{loader_id}/requirements.txt")
        if response.status_code == 200:
            with open(requirements_path, "w") as f:
                f.write(response.text)

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

    spec = util.spec_from_file_location("custom_loader", location=loader_path)
    if spec is None:
        raise ValueError(f"Could not find file: {loader_path}.")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    return getattr(module, loader_class)
