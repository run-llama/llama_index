import os
from pathlib import Path
from typing import List, Optional, Tuple

import requests


def get_file_content(url: str, path: str) -> Tuple[str, int]:
    """Get the content of a file from the GitHub REST API."""
    resp = requests.get(url + path)
    return resp.text, resp.status_code


def get_file_content_bytes(url: str, path: str) -> Tuple[bytes, int]:
    """Get the content of a file from the GitHub REST API."""
    resp = requests.get(url + path)
    return resp.content, resp.status_code


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


def get_source_files_list(source_tree_url: str, path: str) -> List[str]:
    """Get the list of source files to download."""
    resp = requests.get(
        source_tree_url + path + "?recursive=1", headers={"Accept": "application/json"}
    )
    payload = resp.json()["payload"]
    return [item["name"] for item in payload["tree"]["items"]]


def recursive_tree_traverse(
    tree_urls: List[Tuple[str, str]], acc: List[str], source_tree_url: str
):
    """Recursively traversge Github trees to get all file paths in a folder."""
    if not tree_urls:
        return acc
    else:
        url = tree_urls[0]

        try:
            res = requests.get(url, headers={"Accept": "application/json"})
            tree_elements = res.json()["payload"]["tree"]["items"]
        except Exception:
            raise ValueError("Failed to traverse github tree source.")

        new_trees = [
            source_tree_url + "/" + el["path"]
            for el in tree_elements
            if el["contentType"] == "directory"
        ]

        acc += [
            el["path"].replace("llama-index-packs/", "/")
            for el in tree_elements
            if el["contentType"] == "file"
        ]

        return recursive_tree_traverse(
            tree_urls=tree_urls[1:] + new_trees,
            acc=acc,
            source_tree_url=source_tree_url,
        )


def get_source_files_recursive(source_tree_url: str, path: str) -> List[str]:
    """Get source files of a Github folder recursively."""
    initial_url = source_tree_url + path + "?recursive=1"
    initial_tree_urls = [initial_url]
    return recursive_tree_traverse(initial_tree_urls, [], source_tree_url)


class ChangeDirectory:
    """Context manager for changing the current working directory."""

    def __init__(self, new_path: str):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self) -> None:
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback) -> None:
        os.chdir(self.saved_path)
