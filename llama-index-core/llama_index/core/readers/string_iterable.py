"""Simple reader that turns an iterable of strings into a list of Documents."""

from typing import List

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class StringIterableReader(BasePydanticReader):
    """
    String Iterable Reader.

    Gets a list of documents, given an iterable (e.g. list) of strings.

    Example:
        .. code-block:: python

            from llama_index.core.legacy import StringIterableReader, TreeIndex

            documents = StringIterableReader().load_data(
                texts=["I went to the store", "I bought an apple"]
            )
            index = TreeIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            query_engine.query("what did I buy?")

            # response should be something like "You bought an apple."

    """

    is_remote: bool = False

    @classmethod
    def class_name(cls) -> str:
        return "StringIterableReader"

    def load_data(self, texts: List[str]) -> List[Document]:
        """Load the data."""
        results = []
        for text in texts:
            results.append(Document(text=text))

        return results
