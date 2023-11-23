"""Download."""

import json
import os
import subprocess
import sys
from importlib import util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import pkg_resources
import requests
from pkg_resources import DistributionNotFound

LLAMA_HUB_CONTENTS_URL = "https://raw.githubusercontent.com/run-llama/llama-hub/main"
LLAMA_HUB_PATH = "/llama_hub"
LLAMA_HUB_URL = LLAMA_HUB_CONTENTS_URL + LLAMA_HUB_PATH

PATH_TYPE = Union[str, Path]


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


def rewrite_exports(exports: List[str], dirpath: str) -> None:
    """Write the `__all__` variable to the `__init__.py` file in the modules dir.

    Removes the line that contains `__all__` and appends a new line with the updated
    `__all__` variable.

    Args:
        - exports: A list of exported class names.

    """
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


def initialize_directory(
    custom_path: Optional[str] = None, custom_dir: Optional[str] = None
) -> Path:
    """Initialize directory."""
    if custom_path is not None and custom_dir is not None:
        raise ValueError(
            "You cannot specify both `custom_path` and `custom_dir` at the same time."
        )

    custom_dir = custom_dir or "llamahub_modules"
    if custom_path is not None:
        dirpath = Path(custom_path)
    else:
        dirpath = Path(__file__).parent / custom_dir
    if not os.path.exists(dirpath):
        # Create a new directory because it does not exist
        os.makedirs(dirpath)
    if not os.path.exists(f"{dirpath}/__init__.py"):
        # Create an empty __init__.py file if it does not exist yet
        with open(f"{dirpath}/__init__.py", "w") as f:
            pass

    return dirpath


def get_module_info(
    local_dir_path: PATH_TYPE,
    remote_dir_path: PATH_TYPE,
    module_class: str,
    refresh_cache: bool = False,
    library_path: str = "library.json",
    disable_library_cache: bool = False,
) -> Dict:
    """Get module info."""
    if isinstance(local_dir_path, str):
        local_dir_path = Path(local_dir_path)

    local_library_path = f"{local_dir_path}/{library_path}"
    module_id = None  # e.g. `web/simple_web`
    extra_files = []  # e.g. `web/simple_web/utils.py`

    # Check cache first
    if not refresh_cache and os.path.exists(local_library_path):
        with open(local_library_path) as f:
            library = json.load(f)
        if module_class in library:
            module_id = library[module_class]["id"]
            extra_files = library[module_class].get("extra_files", [])

    # Fetch up-to-date library from remote repo if module_id not found
    if module_id is None:
        library_raw_content, _ = _get_file_content(
            str(remote_dir_path), f"/{library_path}"
        )
        library = json.loads(library_raw_content)
        if module_class not in library:
            raise ValueError("Loader class name not found in library")

        module_id = library[module_class]["id"]
        extra_files = library[module_class].get("extra_files", [])

        # create cache dir if needed
        local_library_dir = os.path.dirname(local_library_path)
        if not disable_library_cache:
            if not os.path.exists(local_library_dir):
                os.makedirs(local_library_dir)

            # Update cache
            with open(local_library_path, "w") as f:
                f.write(library_raw_content)

    if module_id is None:
        raise ValueError("Loader class name not found in library")

    return {
        "module_id": module_id,
        "extra_files": extra_files,
    }


def download_module_and_reqs(
    local_dir_path: PATH_TYPE,
    remote_dir_path: PATH_TYPE,
    module_id: str,
    extra_files: List[str],
    refresh_cache: bool = False,
    use_gpt_index_import: bool = False,
    base_file_name: str = "base.py",
    override_path: bool = False,
) -> None:
    """Load module."""
    if isinstance(local_dir_path, str):
        local_dir_path = Path(local_dir_path)

    if override_path:
        module_path = str(local_dir_path)
    else:
        module_path = f"{local_dir_path}/{module_id}"

    if refresh_cache or not os.path.exists(module_path):
        os.makedirs(module_path, exist_ok=True)

        basepy_raw_content, _ = _get_file_content(
            str(remote_dir_path), f"/{module_id}/{base_file_name}"
        )
        if use_gpt_index_import:
            basepy_raw_content = basepy_raw_content.replace(
                "import llama_index", "import llama_index"
            )
            basepy_raw_content = basepy_raw_content.replace(
                "from llama_index", "from llama_index"
            )

        with open(f"{module_path}/{base_file_name}", "w") as f:
            f.write(basepy_raw_content)

        # Get content of extra files if there are any
        # and write them under the loader directory
        for extra_file in extra_files:
            extra_file_raw_content, _ = _get_file_content(
                str(remote_dir_path), f"/{module_id}/{extra_file}"
            )
            # If the extra file is an __init__.py file, we need to
            # add the exports to the __init__.py file in the modules directory
            if extra_file == "__init__.py":
                loader_exports = get_exports(extra_file_raw_content)
                existing_exports = []
                with open(local_dir_path / "__init__.py", "r+") as f:
                    f.write(f"from .{module_id} import {', '.join(loader_exports)}")
                    existing_exports = get_exports(f.read())
                rewrite_exports(existing_exports + loader_exports, str(local_dir_path))
            with open(f"{module_path}/{extra_file}", "w") as f:
                f.write(extra_file_raw_content)

    # install requirements
    requirements_path = f"{local_dir_path}/requirements.txt"

    if not os.path.exists(requirements_path):
        # NOTE: need to check the status code
        response_txt, status_code = _get_file_content(
            str(remote_dir_path), f"/{module_id}/requirements.txt"
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


def download_llama_module(
    module_class: str,
    llama_hub_url: str = LLAMA_HUB_URL,
    refresh_cache: bool = False,
    custom_dir: Optional[str] = None,
    custom_path: Optional[str] = None,
    library_path: str = "library.json",
    base_file_name: str = "base.py",
    use_gpt_index_import: bool = False,
    disable_library_cache: bool = False,
    override_path: bool = False,
) -> Type:
    """Download a module from LlamaHub.

    Can be a loader, tool, pack, or more.

    Args:
        loader_class: The name of the llama module class you want to download,
            such as `GmailOpenAIAgentPack`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        custom_dir: Custom dir name to download loader into (under parent folder).
        custom_path: Custom dirpath to download loader into.
        library_path: File name of the library file.
        use_gpt_index_import: If true, the loader files will use
            llama_index as the base dependency. By default (False),
            the loader files use llama_index as the base dependency.
            NOTE: this is a temporary workaround while we fully migrate all usages
            to llama_index.

    Returns:
        A Loader.
    """
    # create directory / get path
    dirpath = initialize_directory(custom_path=custom_path, custom_dir=custom_dir)

    # fetch info from library.json file
    module_info = get_module_info(
        local_dir_path=dirpath,
        remote_dir_path=llama_hub_url,
        module_class=module_class,
        refresh_cache=refresh_cache,
        library_path=library_path,
        disable_library_cache=disable_library_cache,
    )
    module_id = module_info["module_id"]
    extra_files = module_info["extra_files"]

    # download the module, install requirements
    download_module_and_reqs(
        local_dir_path=dirpath,
        remote_dir_path=llama_hub_url,
        module_id=module_id,
        extra_files=extra_files,
        refresh_cache=refresh_cache,
        use_gpt_index_import=use_gpt_index_import,
        base_file_name=base_file_name,
        override_path=override_path,
    )

    # loads the module into memory
    if override_path:
        spec = util.spec_from_file_location(
            "custom_module", location=f"{dirpath}/{base_file_name}"
        )
        if spec is None:
            raise ValueError(f"Could not find file: {dirpath}/{base_file_name}.")
    else:
        spec = util.spec_from_file_location(
            "custom_module", location=f"{dirpath}/{module_id}/{base_file_name}"
        )
        if spec is None:
            raise ValueError(
                f"Could not find file: {dirpath}/{module_id}/{base_file_name}."
            )

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    return getattr(module, module_class)
