"""Download."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import tqdm

LLAMA_DATASETS_CONTENTS_URL = (
    f"https://raw.githubusercontent.com/run-llama/llama_datasets/main"
)
LLAMA_DATASETS_PATH = "/llama_datasets"
LLAMA_DATASETS_URL = LLAMA_DATASETS_CONTENTS_URL + LLAMA_DATASETS_PATH

LLAMA_DATASETS_LFS_URL = (
    f"https://media.githubusercontent.com/media/run-llama/llama_datasets/main"
)

LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL = (
    "https://github.com/run-llama/llama_datasets/tree/main"
)
LLAMA_RAG_DATASET_FILENAME = "rag_dataset.json"
LLAMA_SOURCE_FILES_PATH = "source_files"


PATH_TYPE = Union[str, Path]


def _get_file_content(loader_hub_url: str, path: str) -> Tuple[str, int]:
    """Get the content of a file from the GitHub REST API."""
    resp = requests.get(loader_hub_url + path)
    return resp.text, resp.status_code


def _get_file_content_bytes(loader_hub_url: str, path: str) -> Tuple[bytes, int]:
    """Get the content of a file from the GitHub REST API."""
    resp = requests.get(loader_hub_url + path)
    return resp.content, resp.status_code


def _get_source_files_list(source_tree_url: str, path: str) -> List[str]:
    """Get the list of source files to download."""
    resp = requests.get(source_tree_url + path + "?recursive=1")
    payload = resp.json()["payload"]
    return [item["name"] for item in payload["tree"]["items"]]


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

    custom_dir = custom_dir or "llamadatasets"
    if custom_path is not None:
        dirpath = Path(custom_path)
    else:
        dirpath = Path(__file__).parent / custom_dir
    if not os.path.exists(dirpath):
        # Create a new directory because it does not exist
        os.makedirs(dirpath)

    return dirpath


def get_dataset_info(
    local_dir_path: PATH_TYPE,
    remote_dir_path: PATH_TYPE,
    remote_source_dir_path: PATH_TYPE,
    dataset_class: str,
    refresh_cache: bool = False,
    library_path: str = "library.json",
    source_files_path: str = "source_files",
    disable_library_cache: bool = False,
) -> Dict:
    """Get dataset info."""
    if isinstance(local_dir_path, str):
        local_dir_path = Path(local_dir_path)

    local_library_path = f"{local_dir_path}/{library_path}"
    dataset_id = None
    source_files = []

    # Check cache first
    if not refresh_cache and os.path.exists(local_library_path):
        with open(local_library_path) as f:
            library = json.load(f)
        if dataset_class in library:
            dataset_id = library[dataset_class]["id"]
            source_files = library[dataset_class].get("source_files", [])

    # Fetch up-to-date library from remote repo if dataset_id not found
    if dataset_id is None:
        library_raw_content, _ = _get_file_content(
            str(remote_dir_path), f"/{library_path}"
        )
        library = json.loads(library_raw_content)
        if dataset_class not in library:
            raise ValueError("Loader class name not found in library")

        dataset_id = library[dataset_class]["id"]
        source_files = _get_source_files_list(
            str(remote_source_dir_path), f"/{dataset_id}/{source_files_path}"
        )

        # create cache dir if needed
        local_library_dir = os.path.dirname(local_library_path)
        if not disable_library_cache:
            if not os.path.exists(local_library_dir):
                os.makedirs(local_library_dir)

            # Update cache
            with open(local_library_path, "w") as f:
                f.write(library_raw_content)

    if dataset_id is None:
        raise ValueError("Dataset class name not found in library")

    return {
        "dataset_id": dataset_id,
        "source_files": source_files,
    }


def download_dataset_and_source_files(
    local_dir_path: PATH_TYPE,
    remote_lfs_dir_path: PATH_TYPE,
    source_files_dir_path: PATH_TYPE,
    dataset_id: str,
    source_files: List[str],
    refresh_cache: bool = False,
    base_file_name: str = "rag_dataset.json",
    override_path: bool = False,
    show_progress: bool = False,
) -> None:
    """Download dataset and source files."""
    if isinstance(local_dir_path, str):
        local_dir_path = Path(local_dir_path)

    if override_path:
        module_path = str(local_dir_path)
    else:
        module_path = f"{local_dir_path}/{dataset_id}"

    if refresh_cache or not os.path.exists(module_path):
        os.makedirs(module_path, exist_ok=True)
        os.makedirs(f"{module_path}/{source_files_dir_path}", exist_ok=True)

        rag_dataset_raw_content, _ = _get_file_content(
            str(remote_lfs_dir_path), f"/{dataset_id}/{base_file_name}"
        )

        with open(f"{module_path}/{base_file_name}", "w") as f:
            f.write(rag_dataset_raw_content)

        # Get content of source files
        if show_progress:
            source_files_iterator = tqdm.tqdm(source_files)
        else:
            source_files_iterator = source_files
        for source_file in source_files_iterator:
            if ".pdf" in source_file:
                source_file_raw_content_bytes, _ = _get_file_content_bytes(
                    str(remote_lfs_dir_path),
                    f"/{dataset_id}/{source_files_dir_path}/{source_file}",
                )
                with open(
                    f"{module_path}/{source_files_dir_path}/{source_file}", "wb"
                ) as f:
                    f.write(source_file_raw_content_bytes)
            else:
                source_file_raw_content, _ = _get_file_content(
                    str(remote_lfs_dir_path),
                    f"/{dataset_id}/{source_files_dir_path}/{source_file}",
                )
                with open(
                    f"{module_path}/{source_files_dir_path}/{source_file}", "w"
                ) as f:
                    f.write(source_file_raw_content)


def download_llama_dataset(
    dataset_class: str,
    llama_datasets_url: str = LLAMA_DATASETS_URL,
    llama_datasets_lfs_url: str = LLAMA_DATASETS_LFS_URL,
    llama_datasets_source_files_tree_url: str = LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    refresh_cache: bool = False,
    custom_dir: Optional[str] = None,
    custom_path: Optional[str] = None,
    source_files_dirpath: str = LLAMA_SOURCE_FILES_PATH,
    library_path: str = "library.json",
    base_file_name: str = "rag_dataset.json",
    disable_library_cache: bool = False,
    override_path: bool = False,
    show_progress: bool = False,
) -> Any:
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
        is_dataset: whether or not downloading a LlamaDataset

    Returns:
        A Loader, A Pack, An Agent, or A Dataset
    """
    # create directory / get path
    dirpath = initialize_directory(custom_path=custom_path, custom_dir=custom_dir)

    # fetch info from library.json file
    dataset_info = get_dataset_info(
        local_dir_path=dirpath,
        remote_dir_path=llama_datasets_url,
        remote_source_dir_path=llama_datasets_source_files_tree_url,
        dataset_class=dataset_class,
        refresh_cache=refresh_cache,
        library_path=library_path,
        disable_library_cache=disable_library_cache,
    )
    dataset_id = dataset_info["dataset_id"]
    source_files = dataset_info["source_files"]

    download_dataset_and_source_files(
        local_dir_path=dirpath,
        remote_lfs_dir_path=llama_datasets_lfs_url,
        source_files_dir_path=source_files_dirpath,
        dataset_id=dataset_id,
        source_files=source_files,
        refresh_cache=refresh_cache,
        base_file_name=base_file_name,
        override_path=override_path,
        show_progress=show_progress,
    )

    if override_path:
        module_path = str(dirpath)
    else:
        module_path = f"{dirpath}/{dataset_id}"

    return (
        f"{module_path}/{LLAMA_RAG_DATASET_FILENAME}",
        f"{module_path}/{LLAMA_SOURCE_FILES_PATH}",
    )
