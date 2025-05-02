"""Download llama-pack as template."""

import logging
import os
import subprocess
import sys
from importlib import util
from pathlib import Path
from typing import Any, Optional, Union

import requests

from llama_index.core.download.utils import (
    ChangeDirectory,
    get_file_content,
    initialize_directory,
    get_source_files_recursive,
)

LLAMA_PACKS_CONTENTS_URL = (
    "https://raw.githubusercontent.com/run-llama/llama_index/main/llama-index-packs"
)
LLAMA_PACKS_SOURCE_FILES_GITHUB_TREE_URL = (
    "https://github.com/run-llama/llama_index/tree/main"
)
PY_NAMESPACE = "llama_index/packs"

PATH_TYPE = Union[str, Path]
LLAMAHUB_ANALYTICS_PROXY_SERVER = "https://llamahub.ai/api/analytics/downloads"

logger = logging.getLogger(__name__)


def download_module_and_reqs(
    local_dir_path: PATH_TYPE,
    remote_dir_path: PATH_TYPE,
    remote_source_dir_path: PATH_TYPE,
    package: str,
    sub_module: str,
    refresh_cache: bool = False,
) -> None:
    """Load module."""
    if isinstance(local_dir_path, str):
        local_dir_path = Path(local_dir_path)

    module_path = f"{local_dir_path}/{PY_NAMESPACE}/{sub_module}"

    if refresh_cache or not os.path.exists(module_path):
        os.makedirs(module_path, exist_ok=True)

        # download all source files
        source_files = get_source_files_recursive(
            str(remote_source_dir_path),
            f"/llama-index-packs/{package}/{PY_NAMESPACE}/{sub_module}",
        )

        for source_file in source_files:
            source_file_raw_content, _ = get_file_content(
                str(remote_dir_path),
                f"{source_file}",
            )
            local_source_file_path = (
                f"{local_dir_path}/{'/'.join(source_file.split('/')[2:])}"
            )
            # ensure parent dir of file exists
            Path(local_source_file_path).parent.absolute().mkdir(
                parents=True, exist_ok=True
            )
            with open(local_source_file_path, "w") as f:
                f.write(source_file_raw_content)

    # pyproject.toml and README
    pyproject_toml_path = f"{local_dir_path}/pyproject.toml"
    readme_path = (
        f"{local_dir_path}/README.md"  # needed to install deps from pyproject.toml
    )

    if not os.path.exists(pyproject_toml_path):
        # NOTE: need to check the status code
        response_txt, status_code = get_file_content(
            str(remote_dir_path), f"/{package}/pyproject.toml"
        )
        if status_code == 200:
            with open(pyproject_toml_path, "w") as f:
                f.write(response_txt)

    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(
                "DO NOT DELETE\nThis readme file is needed to install from pyproject.toml."
            )

    # Install dependencies
    if os.path.exists(pyproject_toml_path):
        with ChangeDirectory(str(local_dir_path)):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "."])


def download_llama_pack_template(
    new_install_parent: str,
    llama_pack_class: str,
    llama_pack_url: str = LLAMA_PACKS_CONTENTS_URL,
    llama_pack_source_files_dir_path: str = LLAMA_PACKS_SOURCE_FILES_GITHUB_TREE_URL,
    refresh_cache: bool = False,
    custom_dir: Optional[str] = None,
    custom_path: Optional[str] = None,
    base_file_name: str = "__init__.py",
) -> Any:
    # create directory / get path
    dirpath = initialize_directory(custom_path=custom_path, custom_dir=custom_dir)

    sub_module = new_install_parent.replace("llama-index-packs-", "").replace("-", "_")

    # download the module, install requirements
    download_module_and_reqs(
        local_dir_path=dirpath,
        remote_dir_path=llama_pack_url,
        remote_source_dir_path=llama_pack_source_files_dir_path,
        package=new_install_parent,
        sub_module=sub_module,
        refresh_cache=refresh_cache,
    )

    # loads the module into memory
    path = f"{dirpath}/{PY_NAMESPACE}/{sub_module}/{base_file_name}"
    spec = util.spec_from_file_location("llama_index.packs._custom", location=path)
    if spec is None:
        raise ValueError(f"Could not find file: {path}.")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    return getattr(module, llama_pack_class)


def track_download(module_class: str, module_type: str) -> None:
    """Tracks number of downloads via Llamahub proxy.

    Args:
        module_class: The name of the llama module being downloaded, e.g.,`GmailOpenAIAgentPack`.
        module_type: Can be "loader", "tool", "llamapack", or "datasets"
    """
    try:
        requests.post(
            LLAMAHUB_ANALYTICS_PROXY_SERVER,
            json={"type": module_type, "plugin": module_class},
        )
    except Exception as e:
        logger.info(f"Error tracking downloads for {module_class} : {e}")
