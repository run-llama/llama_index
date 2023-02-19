"""Download loader from the Loader Hub."""

import json
import os
import subprocess
import sys
from importlib import util
from pathlib import Path
from typing import Optional

import pkg_resources
import requests
from pkg_resources import DistributionNotFound

from gpt_index.readers.base import BaseReader

LOADER_HUB_URL = (
    "https://raw.githubusercontent.com/emptycrown/loader-hub/main/loader_hub"
)


def download_loader(
    loader_class: str,
    loader_hub_url: str = LOADER_HUB_URL,
    refresh_cache: Optional[bool] = False,
) -> BaseReader:
    """Download a single loader from the Loader Hub.

    Args:
        loader_class: The name of the loader class you want to download,
            such as `SimpleWebPageReader`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.

    Returns:
        A Loader.
    """
    dirpath = ".modules"
    if not os.path.exists(dirpath):
        # Create a new directory because it does not exist
        os.makedirs(dirpath)

    library_path = f"{dirpath}/library.json"
    loader_id = None  # e.g. `web/simple_web`

    # Check cache first
    if not refresh_cache and os.path.exists(library_path):
        with open(library_path) as f:
            library = json.load(f)
        if loader_class in library:
            loader_id = library[loader_class]["id"]

    # Fetch up-to-date library from remote repo if loader_id not found
    if loader_id is None:
        response = requests.get(f"{loader_hub_url}/library.json")
        library = json.loads(response.text)
        if loader_class not in library:
            raise ValueError("Loader class name not found in library")

        loader_id = library[loader_class]["id"]
        # Update cache
        with open(library_path, "w") as f:
            f.write(response.text)

    assert loader_id is not None
    # Load the module
    loader_filename = loader_id.replace("/", "-")
    loader_path = f"{dirpath}/{loader_filename}.py"
    requirements_path = f"{dirpath}/{loader_filename}_requirements.txt"

    if refresh_cache or not os.path.exists(loader_path):
        response = requests.get(f"{loader_hub_url}/{loader_id}/base.py")
        with open(loader_path, "w") as f:
            f.write(response.text)

    if not os.path.exists(requirements_path):
        response = requests.get(f"{loader_hub_url}/{loader_id}/requirements.txt")
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
