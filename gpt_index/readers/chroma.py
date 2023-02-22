"""Chroma Reader."""

from typing import Any

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class ChromaReader(BaseReader):
    """Chroma reader.

    Retrieve documents from existing persisted Chroma collections.

    Args:
        collection_name: Name of the peristed collection.
        persist_directory: Directory where the collection is persisted.

    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
    ) -> None:
        """Initialize with parameters."""
        import_err_msg = (
            "`chromadb` package not found, please run `pip install chromadb`"
        )
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        if (collection_name is None) or (persist_directory is None):
            raise ValueError("Please provide a collection name and persist directory.")

        from chromadb.config import Settings

        self._client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=persist_directory
            )
        )
        self._collection = self._client.get_collection(collection_name)

    def load_data(
        self,
        query_vector: Any,
        limit: int = 10,
    ) -> Any:
        """Load data from Chroma.

        Args:
            query_vector (Any): Query
            limit (int): Number of results to return.

        Returns:
            List[Document]: A list of documents.
        """
        results = self._collection.query(query_embeddings=query_vector, n_results=limit)

        documents = []
        for result in zip(results["ids"], results["documents"], results["embeddings"]):
            document = Document(
                doc_id=result[0][0],
                text=result[1][0],
                embedding=result[2][0],
            )
            documents.append(document)

        return documents
