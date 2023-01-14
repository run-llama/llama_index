"""Base reader class."""
from abc import abstractmethod
from typing import Any, List

from langchain.docstore.document import Document as LCDocument

from gpt_index.readers.schema.base import Document


class BaseReader:
    """Utilities for loading data from a directory."""

    def __init__(self, verbose: bool = False):
        """Init params."""
        self.verbose = verbose

    @abstractmethod
    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""

    def load_langchain_documents(self, **load_kwargs: Any) -> List[LCDocument]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]
