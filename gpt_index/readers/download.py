"""Download loader from the Loader Hub."""

import json
import os
from importlib import util

from langchain.utilities import RequestsWrapper

from gpt_index.readers.base import BaseReader

LOADER_HUB_URL = (
    "https://raw.githubusercontent.com/emptycrown/loader-hub/main/loader_hub"
)


def download_loader(loaderClassName: str) -> BaseReader:
    """Download a single loader from the Loader Hub.

    Args:
        loaderClassName: The name of the loader class you want to download,
            such as `SimpleWebPageReader`.
    Returns:
        A Loader.
    """
    requests = RequestsWrapper()
    response = requests.run(f"{LOADER_HUB_URL}/library.json")
    library = json.loads(response)

    # Look up the loader id (e.g. `web/simple_web`)
    loader_id = library[loaderClassName]["id"]
    dirpath = ".modules"
    loader_filename = loader_id.replace("/", "-")
    loader_path = f"{dirpath}/{loader_filename}.py"

    if not os.path.exists(dirpath):
        # Create a new directory because it does not exist
        os.makedirs(dirpath)

    if not os.path.exists(loader_path):
        response = requests.run(f"{LOADER_HUB_URL}/{loader_id}/base.py")
        with open(loader_path, "w") as f:
            f.write(response)

    spec = util.spec_from_file_location("custom_loader", location=loader_path)
    if spec is None:
        raise ValueError(f"Could not find file: {loader_path}.")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    return getattr(module, loaderClassName)
