"""Simple reader that ."""
from pathlib import Path
from typing import Any, List

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class SimpleDirectoryReader(BaseReader):
    """Simple directory reader.

    Can read files into separate documents, or concatenates
    files into one document text.

    Args:
        input_dir (str): Path to the directory.
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open

    """

    def __init__(
        self, input_dir: str, exclude_hidden: bool = True, errors: str = "ignore"
    ) -> None:
        """Initialize with parameters."""
        self.input_dir = Path(input_dir)
        self.errors = errors
        input_files = list(self.input_dir.iterdir())
        if exclude_hidden:
            input_files = [f for f in input_files if not f.name.startswith(".")]
        for input_file in input_files:
            if not input_file.is_file():
                raise ValueError(f"Expected {input_file} to be a file.")
        self.input_files = input_files

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        Args:
            concatenate (bool): whether to concatenate all files into one document.

        Returns:
            List[Document]: A list of documents.

        """
        concatenate = load_kwargs.get("concatenate", True)
        data = ""
        data_list = []
        for input_file in self.input_files:
            with open(input_file, "r", errors=self.errors) as f:
                data = f.read()
                data_list.append(data)

        if concatenate:
            return [Document("\n".join(data_list))]
        else:
            return [Document(d) for d in data_list]
