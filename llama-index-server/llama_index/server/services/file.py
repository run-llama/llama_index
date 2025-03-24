import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PRIVATE_STORE_PATH = str(Path("output", "uploaded"))
TOOL_STORE_PATH = str(Path("output", "tools"))
LLAMA_CLOUD_STORE_PATH = str(Path("output", "llamacloud"))


class DocumentFile(BaseModel):
    id: str
    name: str  # Stored file name
    type: Optional[str] = None
    size: Optional[int] = None
    url: Optional[str] = None
    path: Optional[str] = Field(
        None,
        description="The stored file path. Used internally in the server.",
        exclude=True,
    )
    refs: Optional[List[str]] = Field(
        None, description="The document ids in the index."
    )


class FileService:
    """
    To store the files uploaded by the user.
    """

    @classmethod
    def save_file(
        cls,
        content: Union[bytes, str],
        file_name: str,
        save_dir: Optional[str] = None,
    ) -> DocumentFile:
        """
        Save the content to a file in the local file server (accessible via URL).

        Args:
            content (bytes | str): The content to save, either bytes or string.
            file_name (str): The original name of the file.
            save_dir (Optional[str]): The relative path from the current working directory. Defaults to the `output/uploaded` directory.

        Returns:
            The metadata of the saved file.
        """
        if save_dir is None:
            save_dir = os.path.join("output", "uploaded")

        file_id = str(uuid.uuid4())
        name, extension = os.path.splitext(file_name)
        extension = extension.lstrip(".")
        sanitized_name = _sanitize_file_name(name)
        if extension == "":
            raise ValueError("File is not supported!")
        new_file_name = f"{sanitized_name}_{file_id}.{extension}"

        file_path = os.path.join(save_dir, new_file_name)

        if isinstance(content, str):
            content = content.encode()

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file:
                file.write(content)
        except PermissionError as e:
            logger.error(f"Permission denied when writing to file {file_path}: {e!s}")
            raise
        except OSError as e:
            logger.error(f"IO error occurred when writing to file {file_path}: {e!s}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when writing to file {file_path}: {e!s}")
            raise

        logger.info(f"Saved file to {file_path}")

        file_url_prefix = os.getenv("FILESERVER_URL_PREFIX")
        if file_url_prefix is None:
            logger.warning(
                "FILESERVER_URL_PREFIX is not set. Some features may not work correctly."
            )
            file_url_prefix = "http://localhost:8000/api/files"
        file_size = os.path.getsize(file_path)

        file_url = os.path.join(
            file_url_prefix,
            save_dir,
            new_file_name,
        )

        return DocumentFile(
            id=file_id,
            name=new_file_name,
            type=extension,
            size=file_size,
            path=file_path,
            url=file_url,
            refs=None,
        )


def _sanitize_file_name(file_name: str) -> str:
    """
    Sanitize the file name by replacing all non-alphanumeric characters with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9.]", "_", file_name)
