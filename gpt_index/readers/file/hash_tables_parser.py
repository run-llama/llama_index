"""Hash tables parser.
Contains parsers for json files.
"""

from pathlib import Path
from typing import Dict

from gpt_index.readers.file.base_parser import BaseParser

class JSONParser(BaseParser):
    """JSON parser."""

    def _init_parser(self) -> Dict:
        """Init parser."""
        return {}

    def parse_file(self, file: Path, errors: str = "ignore") -> str:
        """Parse file."""
        try:
            import json
        except ImportError:
            raise ValueError("json is required to read JSON files.")
        with open(file, "r") as f:
            json_data = json.load(f)
            text = json.dumps(json_data)
        return text
