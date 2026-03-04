"""Astra DB."""

from typing import Any, List, Optional

import llama_index.core
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class AstraDBReader(BaseReader):
    """
    Astra DB reader.

    Retrieve documents from an Astra DB Instance.

    Args:
        collection_name (str): collection name to use. If not existing, it will be created.
        token (str): The Astra DB Application Token to use.
        api_endpoint (str): The Astra DB JSON API endpoint for your database.
        embedding_dimension (int): Length of the embedding vectors in use.
        namespace (Optional[str]): The namespace to use. If not provided, 'default_keyspace'
        client (Optional[Any]): Astra DB client to use. If not provided, one will be created.

    """

    def __init__(
        self,
        *,
        collection_name: str,
        token: str,
        api_endpoint: str,
        embedding_dimension: int,
        namespace: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize with parameters."""
        import_err_msg = (
            "`astrapy` package not found, please run `pip install --upgrade astrapy`"
        )

        # Try to import astrapy for use
        try:
            from astrapy.db import AstraDB
        except ImportError:
            raise ImportError(import_err_msg)

        if client is not None:
            self._client = client.copy()
            self._client.set_caller(
                caller_name=getattr(llama_index, "__name__", "llama_index"),
                caller_version=getattr(llama_index.core, "__version__", None),
            )
        else:
            # Build the Astra DB object
            self._client = AstraDB(
                api_endpoint=api_endpoint,
                token=token,
                namespace=namespace,
                caller_name=getattr(llama_index, "__name__", "llama_index"),
                caller_version=getattr(llama_index.core, "__version__", None),
            )

        self._collection = self._client.create_collection(
            collection_name=collection_name, dimension=embedding_dimension
        )

    def load_data(self, vector: List[float], limit: int = 10, **kwargs: Any) -> Any:
        """
        Load data from Astra DB.

        Args:
            vector (Any): Query
            limit (int): Number of results to return.
            kwargs (Any): Additional arguments to pass to the Astra DB query.

        Returns:
            List[Document]: A list of documents.

        """
        results = self._collection.vector_find(
            vector,
            limit=limit,
            fields=["*"],
            **kwargs,
        )

        documents: List[Document] = []
        for result in results:
            document = Document(
                doc_id=result["_id"],
                text=result["content"],
                embedding=result["$vector"],
            )

            documents.append(document)

        return documents
