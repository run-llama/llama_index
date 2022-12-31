"""Simple reader that ."""
from pathlib import Path
from typing import Any, List, Optional

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
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        required_exts (Optional[List[str]]): List of required extensions.
            Default is None.

    """

    def __init__(
        self,
        input_dir: str,
        exclude_hidden: bool = True,
        errors: str = "ignore",
        recursive: bool = False,
        required_exts: Optional[List[str]] = None,
    ) -> None:
        """Initialize with parameters."""
        self.input_dir = Path(input_dir)
        self.errors = errors

        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.required_exts = required_exts

        self.input_files = self._add_files(self.input_dir)

    def _add_files(self, input_dir: Path) -> List[Path]:
        """Add files."""
        input_files = sorted(input_dir.iterdir())
        new_input_files = []
        for input_file in input_files:
            if input_file.is_dir():
                sub_input_files = self._add_files(input_file)
                new_input_files.extend(sub_input_files)
            elif self.exclude_hidden and input_file.name.startswith("."):
                continue
            elif (
                self.required_exts is not None
                and input_file.suffix not in self.required_exts
            ):
                continue
            else:
                new_input_files.append(input_file)

        return new_input_files

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        Args:
            concatenate (bool): whether to concatenate all files into one document.
                False by default.

        Returns:
            List[Document]: A list of documents.

        """
        concatenate = load_kwargs.get("concatenate", False)
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
