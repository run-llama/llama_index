"""Pinecone reader."""

from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ZepReader(BaseReader):
    """
    Zep document vector store reader.

    Args:
        api_url (str): Zep API URL
        api_key (str): Zep API key, optional

    """

    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """Initialize with parameters."""
        from zep_python import ZepClient

        self._api_url = api_url
        self._api_key = api_key
        self._client = ZepClient(base_url=api_url, api_key=api_key)

    def load_data(
        self,
        collection_name: str,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = 5,
        separate_documents: Optional[bool] = True,
        include_values: Optional[bool] = True,
    ) -> List[Document]:
        """
        Load data from Zep.

        Args:
            collection_name (str): Name of the Zep collection.
            query (Optional[str]): Query string. Required if vector is None.
            vector (Optional[List[float]]): Query vector. Required if query is None.
            metadata (Optional[Dict[str, Any]]): Metadata to filter on.
            top_k (Optional[int]): Number of results to return. Defaults to 5.
            separate_documents (Optional[bool]): Whether to return separate
                documents per retrieved entry. Defaults to True.
            include_values (Optional[bool]): Whether to include the embedding in
                the response. Defaults to True.

        Returns:
            List[Document]: A list of documents.

        """
        if query is None and vector is None:
            raise ValueError("Either query or vector must be specified.")

        collection = self._client.document.get_collection(name=collection_name)
        response = collection.search(
            text=query, embedding=vector, limit=top_k, metadata=metadata
        )

        documents = [
            (
                Document(text=d.content, embedding=d.embedding)
                if include_values
                else Document(text=d.content)
            )
            for d in response
        ]

        if not separate_documents:
            text_list = [d.get_text() for d in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
