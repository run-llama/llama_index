"""Simple reader that ."""
from pathlib import Path
from typing import Any, List

from gpt_index.readers.base import BaseReader
from gpt_index.schema import Document


class SimpleDirectoryReader(BaseReader):
    """Simple directory reader.

    Concatenates all files into one document text.

    """

    def __init__(self, input_dir: str) -> None:
        """Initialize with parameters."""
        self.input_dir = Path(input_dir)
        input_files = list(self.input_dir.iterdir())
        for input_file in input_files:
            if not input_file.is_file():
                raise ValueError(f"Expected {input_file} to be a file.")
        self.input_files = input_files

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        data = ""
        for input_file in self.input_files:
            with open(input_file, "r") as f:
                data += f.read()
            data += "\n"
        return [Document(data)]
