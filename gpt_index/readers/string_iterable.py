from abc import abstractmethod
from typing import Any, Iterable, List

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class StringIterableReader(BaseReader):
    """String Iterable Reader
    
    Gets a list of documents, given an iterable (e.g. list) of strings

    Args:
        texts (Iterable[str]): Iterable of strings to be added to the Document list result.

    Example:
        .. code-block:: python
            from gpt_index import StringIterableReader, GPTTreeIndex

            documents = StringIterableReader(["I went to the store", "I bought an apple"]).load_data()
            index = GPTTreeIndex(documents)
            index.query("what did I buy?")

            # response should be something like "You bought an apple."
    """

    def __init__(self, texts: Iterable[str]):
        """Initialize with parameters."""
        self.texts = texts

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """
        Load the data
        """
        results = []
        for text in self.texts:
            results.append(Document(text))

        return results