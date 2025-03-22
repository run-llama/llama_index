import logging
import shutil
from pathlib import Path
from typing import Optional
import requests

# TODO: Update to npm registry once we have a release of @llamaindex/server
CHAT_UI_VERSION = "1759"


def download_chat_ui(
    logger: Optional[logging.Logger] = None, target_path: str = ".ui"
) -> None:
    if logger is None:
        logger = logging.getLogger("uvicorn")
    path = Path(target_path)
    temp_dir = _download_package(_get_download_link(CHAT_UI_VERSION))
    print("Downloaded chat ui to", temp_dir)
    _copy_ui_files(temp_dir, path)
    logger.info("Chat UI downloaded and copied to static folder")


def _get_download_link(version: str) -> str:
    """
    Get the download link for the chat UI.

    TODO: Update to npm registry once we have a release of @llamaindex/server
    """
    return f"https://pkg.pr.new/run-llama/LlamaIndexTS/@llamaindex/server@{version}"


def _download_package(url: str) -> Path:
    """Download tar.gz file and extract all files into a temporary directory."""
    import io
    import tarfile
    import tempfile

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    content = response.content

    temp_dir = Path(tempfile.mkdtemp())

    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
        tar.extractall(path=temp_dir)

    return temp_dir


def _copy_ui_files(temp_dir: Path, target_path: Path) -> None:
    """Copy files from the .next directory to the static directory."""
    target_path.mkdir(parents=True, exist_ok=True)
    next_dir = temp_dir / "package/dist/static"

    if next_dir.exists():
        for item in next_dir.iterdir():
            dest = target_path / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
