"""Flat reader."""
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class FlatReader(BaseReader):
    """Flat reader.

    Extract raw text from a file and save the file type in the metadata
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)

    def _get_fs(
        self, file: Path, fs: Optional[AbstractFileSystem] = None
    ) -> AbstractFileSystem:
        if fs is None:
            fs = LocalFileSystem()
        return fs

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file into string."""
        fs = self._get_fs(file, fs)
        with fs.open(file, encoding="utf-8") as f:
            content = f.read()
        metadata = {"filename": file.name, "extension": file.suffix}
        if extra_info:
            metadata = {**metadata, **extra_info}

        return [Document(text=content, metadata=metadata)]
