"""Download loader from the Loader Hub.

NOTE: using `download_loader` is now deprecated.
Please do `pip install llama-hub` instead.

"""

import json
import os
import subprocess
import sys
from importlib import util
from pathlib import Path
from typing import List, Optional, Tuple, Type

import pkg_resources
import requests
from pkg_resources import DistributionNotFound

from llama_index.readers.base import BaseReader

LLAMA_HUB_CONTENTS_URL = "https://raw.githubusercontent.com/emptycrown/llama-hub/main"
LOADER_HUB_PATH = "/llama_hub"
LOADER_HUB_URL = LLAMA_HUB_CONTENTS_URL + LOADER_HUB_PATH


def _get_file_content(loader_hub_url: str, path: str) -> Tuple[str, int]:
    """Get the content of a file from the GitHub REST API."""
    resp = requests.get(loader_hub_url + path)
    return resp.text, resp.status_code


def get_exports(raw_content: str) -> List:
    """Read content of a Python file and returns a list of exported class names.

    For example:
    ```python
    from .a import A
    from .b import B

    __all__ = ["A", "B"]
    ```
    will return `["A", "B"]`.

    Args:
        - raw_content: The content of a Python file as a string.

    Returns:
        A list of exported class names.

    """
    exports = []
    for line in raw_content.splitlines():
        line = line.strip()
        if line.startswith("__all__"):
            exports = line.split("=")[1].strip().strip("[").strip("]").split(",")
            exports = [export.strip().strip("'").strip('"') for export in exports]
    return exports


def rewrite_exports(exports: List[str], custom_path: Optional[str] = None) -> None:
    """Write the `__all__` variable to the `__init__.py` file in the modules dir.

    Removes the line that contains `__all__` and appends a new line with the updated
    `__all__` variable.

    Args:
        - exports: A list of exported class names.

    """
    if custom_path is not None:
        dirpath = Path(custom_path)
    else:
        dirpath = Path(__file__).parent / "llamahub_modules"
    init_path = f"{dirpath}/__init__.py"
    with open(init_path) as f:
        lines = f.readlines()
    with open(init_path, "w") as f:
        for line in lines:
            line = line.strip()
            if line.startswith("__all__"):
                continue
            f.write(line + os.linesep)
        f.write(f"__all__ = {list(set(exports))}" + os.linesep)


def download_loader(
    loader_class: str,
    loader_hub_url: str = LOADER_HUB_URL,
    refresh_cache: Optional[bool] = False,
    use_gpt_index_import: bool = False,
    custom_path: Optional[str] = None,
) -> Type[BaseReader]:
    """Download a single loader from the Loader Hub.

    Args:
        loader_class: The name of the loader class you want to download,
            such as `SimpleWebPageReader`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        use_gpt_index_import: If true, the loader files will use
            llama_index as the base dependency. By default (False),
            the loader files use llama_index as the base dependency.
            NOTE: this is a temporary workaround while we fully migrate all usages
            to llama_index.
        custom_path: Custom dirpath to download loader into.

    Returns:
        A Loader.
    """
    if custom_path is not None:
        dirpath = Path(custom_path)
    else:
        dirpath = Path(__file__).parent / "llamahub_modules"
    if not os.path.exists(dirpath):
        # Create a new directory because it does not exist
        os.makedirs(dirpath)
    if not os.path.exists(f"{dirpath}/__init__.py"):
        # Create an empty __init__.py file if it does not exist yet
        with open(f"{dirpath}/__init__.py", "w") as f:
            pass

    library_path = f"{dirpath}/library.json"
    loader_id = None  # e.g. `web/simple_web`
    extra_files = []  # e.g. `web/simple_web/utils.py`

    # Check cache first
    if not refresh_cache and os.path.exists(library_path):
        with open(library_path) as f:
            library = json.load(f)
        if loader_class in library:
            loader_id = library[loader_class]["id"]
            extra_files = library[loader_class].get("extra_files", [])

    # Fetch up-to-date library from remote repo if loader_id not found
    if loader_id is None:
        library_raw_content, _ = _get_file_content(loader_hub_url, "/library.json")
        library = json.loads(library_raw_content)
        if loader_class not in library:
            raise ValueError("Loader class name not found in library")

        loader_id = library[loader_class]["id"]
        extra_files = library[loader_class].get("extra_files", [])
        # Update cache
        with open(library_path, "w") as f:
            f.write(library_raw_content)

    if loader_id is None:
        raise ValueError("Loader class name not found in library")
    # Load the module
    loader_path = f"{dirpath}/{loader_id}"
    requirements_path = f"{loader_path}/requirements.txt"

    if refresh_cache or not os.path.exists(loader_path):
        os.makedirs(loader_path, exist_ok=True)
        basepy_raw_content, _ = _get_file_content(
            loader_hub_url, f"/{loader_id}/base.py"
        )
        if use_gpt_index_import:
            basepy_raw_content = basepy_raw_content.replace(
                "import llama_index", "import llama_index"
            )
            basepy_raw_content = basepy_raw_content.replace(
                "from llama_index", "from llama_index"
            )

        with open(f"{loader_path}/base.py", "w") as f:
            f.write(basepy_raw_content)

        # Get content of extra files if there are any
        # and write them under the loader directory
        for extra_file in extra_files:
            extra_file_raw_content, _ = _get_file_content(
                loader_hub_url, f"/{loader_id}/{extra_file}"
            )
            # If the extra file is an __init__.py file, we need to
            # add the exports to the __init__.py file in the modules directory
            if extra_file == "__init__.py":
                loader_exports = get_exports(extra_file_raw_content)
                existing_exports = []
                with open(dirpath / "__init__.py", "r+") as f:
                    f.write(f"from .{loader_id} import {', '.join(loader_exports)}")
                    existing_exports = get_exports(f.read())
                rewrite_exports(
                    existing_exports + loader_exports, custom_path=custom_path
                )
            with open(f"{loader_path}/{extra_file}", "w") as f:
                f.write(extra_file_raw_content)

    if not os.path.exists(requirements_path):
        # NOTE: need to check the status code
        response_txt, status_code = _get_file_content(
            loader_hub_url, f"/{loader_id}/requirements.txt"
        )
        if status_code == 200:
            with open(requirements_path, "w") as f:
                f.write(response_txt)

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
