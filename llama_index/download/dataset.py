"""Download."""

import json
import os
from pathlib import Path
from typing import Dict, List, Union

import requests
import tqdm

from llama_index.download.utils import (
    get_file_content,
    get_file_content_bytes,
)

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


def _get_source_files_list(source_tree_url: str, path: str) -> List[str]:
    """Get the list of source files to download."""
    resp = requests.get(source_tree_url + path + "?recursive=1")
    payload = resp.json()["payload"]
    return [item["name"] for item in payload["tree"]["items"]]


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
        library_raw_content, _ = get_file_content(
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

        rag_dataset_raw_content, _ = get_file_content(
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
                source_file_raw_content_bytes, _ = get_file_content_bytes(
                    str(remote_lfs_dir_path),
                    f"/{dataset_id}/{source_files_dir_path}/{source_file}",
                )
                with open(
                    f"{module_path}/{source_files_dir_path}/{source_file}", "wb"
                ) as f:
                    f.write(source_file_raw_content_bytes)
            else:
                source_file_raw_content, _ = get_file_content(
                    str(remote_lfs_dir_path),
                    f"/{dataset_id}/{source_files_dir_path}/{source_file}",
                )
                with open(
                    f"{module_path}/{source_files_dir_path}/{source_file}", "w"
                ) as f:
                    f.write(source_file_raw_content)
