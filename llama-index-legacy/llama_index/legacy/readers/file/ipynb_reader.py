import re
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.schema import Document


class IPYNBReader(BaseReader):
    """Image parser."""

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
        concatenate: bool = False,
    ):
        """Init params."""
        self._parser_config = parser_config
        self._concatenate = concatenate

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        if file.name.endswith(".ipynb"):
            try:
                import nbconvert
            except ImportError:
                raise ImportError("Please install nbconvert 'pip install nbconvert' ")
        string = nbconvert.exporters.ScriptExporter().from_file(file)[0]
        # split each In[] cell into a separate string
        splits = re.split(r"In\[\d+\]:", string)
        # remove the first element, which is empty
        splits.pop(0)

        if self._concatenate:
            docs = [Document(text="\n\n".join(splits), metadata=extra_info or {})]
        else:
            docs = [Document(text=s, metadata=extra_info or {}) for s in splits]
        return docs
