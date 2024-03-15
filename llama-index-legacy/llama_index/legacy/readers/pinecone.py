"""Pinecone reader."""

from typing import Any, Dict, List, Optional

from llama_index.legacy.readers.base import BaseReader
from llama_index.legacy.schema import Document


class PineconeReader(BaseReader):
    """Pinecone reader.

    Args:
        api_key (str): Pinecone API key.
        environment (str): Pinecone environment.
    """

    def __init__(self, api_key: str, environment: Optional[str] = None) -> None:
        """Initialize with parameters."""
        raise NotImplementedError(
            "PineconeReader has been deprecated. Please use `PineconeVectorStore` instead."
        )

    def load_data(
        self,
        index_name: str,
        id_to_text_map: Dict[str, str],
        vector: Optional[List[float]],
        top_k: int,
        separate_documents: bool = True,
        include_values: bool = True,
        **query_kwargs: Any
    ) -> List[Document]:
        """Load data from Pinecone.

        Args:
            index_name (str): Name of the index.
            id_to_text_map (Dict[str, str]): A map from ID's to text.
            separate_documents (Optional[bool]): Whether to return separate
                documents per retrieved entry. Defaults to True.
            vector (List[float]): Query vector.
            top_k (int): Number of results to return.
            include_values (bool): Whether to include the embedding in the response.
                Defaults to True.
            **query_kwargs: Keyword arguments to pass to the query.
                Arguments are the exact same as those found in
                Pinecone's reference documentation for the
                query method.

        Returns:
            List[Document]: A list of documents.
        """
        raise NotImplementedError(
            "PineconeReader has been deprecated. Please use `PineconeVectorStore` instead."
        )
