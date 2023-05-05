import re
from pathlib import Path
from typing import Dict, Optional, List

from llama_index.readers.file.base_parser import BaseParser


class IPYNBParser(BaseParser):
    """Image parser."""

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
    ):
        """Init params."""
        self._parser_config = parser_config

    def _init_parser(self) -> Dict:
        """Init parser."""
        return {}

    def parse_file(self, file: Path, errors: str = "ignore") -> List[str]:
        """Parse file."""

        if file.name.endswith(".ipynb"):
            try:
                import nbconvert  # noqa: F401
            except ImportError:
                raise ImportError("Please install nbconvert 'pip install nbconvert' ")
        string = nbconvert.exporters.ScriptExporter().from_file(file)[0]
        # split each In[] cell into a separate string
        split = re.split(r"In\[\d+\]:", string)
        # remove the first element, which is empty
        split.pop(0)
        return split
