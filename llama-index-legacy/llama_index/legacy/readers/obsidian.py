"""Obsidian reader class.

Pass in the path to an Obsidian vault and it will parse all markdown
files into a List of Documents,
with each Document containing text from under an Obsidian header.

"""
import os
from pathlib import Path
from typing import Any, List

from llama_index.readers.base import BaseReader
from llama_index.readers.file.markdown_reader import MarkdownReader
from llama_index.schema import Document


class ObsidianReader(BaseReader):
    """Utilities for loading data from an Obsidian Vault.

    Args:
        input_dir (str): Path to the vault.

    """

    def __init__(self, input_dir: str):
        """Init params."""
        self.input_dir = Path(input_dir)

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        docs: List[Document] = []
        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.endswith(".md"):
                    filepath = os.path.join(dirpath, filename)
                    content = MarkdownReader().load_data(Path(filepath))
                    docs.extend(content)
        return docs
