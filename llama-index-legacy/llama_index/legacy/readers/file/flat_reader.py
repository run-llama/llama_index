"""Flat reader."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.legacy.readers.base import BaseReader
from llama_index.legacy.schema import Document


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

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file into string."""
        with open(file, encoding="utf-8") as f:
            content = f.read()
        metadata = {"filename": file.name, "extension": file.suffix}
        if extra_info:
            metadata = {**metadata, **extra_info}

        return [Document(text=content, metadata=metadata)]
