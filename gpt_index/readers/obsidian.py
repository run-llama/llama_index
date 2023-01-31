"""Base reader class."""
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, List

from langchain.docstore.document import Document as LCDocument

from gpt_index.readers.base import BaseReader
from gpt_index.readers.file.markdown_parser import MarkdownParser
from gpt_index.readers.schema.base import Document


class ObsidianReader(BaseReader):
    """Utilities for loading data from a directory."""

    def __init__(self, input_dir: str, verbose: bool = False):
        """Init params."""
        self.verbose = verbose
        self.input_dir = Path(input_dir)

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        try:
            import re
        except ImportError:
            raise ValueError("re module is required to read Markdown files.")
        docs: List[str] = []
        for (dirpath, dirnames, filenames) in os.walk(self.input_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.endswith(".md"):
                    filepath = os.path.join(dirpath, filename)
                    content = MarkdownParser().parse_file(Path(filepath))
                    pieces = content.values()
                    docs.extend(pieces)
        return [Document(d) for d in docs]

    def load_langchain_documents(self, **load_kwargs: Any) -> List[LCDocument]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]
