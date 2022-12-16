"""Pinecone reader."""

from typing import Any, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.schema import Document


class PineconeReader(BaseReader):
    """Pinecone reader."""

    def __init__(self, api_key: str, environment: str):
        """Initialize with parameters."""
        try:
            import pinecone # noqa: F401
        except ImportError:
            raise ValueError(
                "`pinecone` package not found, please run `pip install pinecone-client`"
            )

        self._api_key = api_key
        self._environment = environment
        pinecone.init(api_key=api_key, environment=environment)


    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from Pinecone.
        
        Args:
            index_name (str): Name of the index.
            query_vecotr

        Returns:
            List[Document]: A list of documents.
        """

        index.