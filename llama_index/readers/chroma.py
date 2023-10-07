"""Chroma Reader."""

from typing import Any, List, Optional, Union

from llama_index.readers.base import BaseReader
from llama_index.schema import Document


class ChromaReader(BaseReader):
    """Chroma reader.

    Retrieve documents from existing persisted Chroma collections.

    Args:
        collection_name: Name of the persisted collection.
        persist_directory: Directory where the collection is persisted.

    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        chroma_api_impl: str = "rest",
        chroma_db_impl: Optional[str] = None,
        host: str = "localhost",
        port: int = 8000,
    ) -> None:
        """Initialize with parameters."""
        import_err_msg = (
            "`chromadb` package not found, please run `pip install chromadb`"
        )
        try:
            import chromadb
        except ImportError:
            raise ImportError(import_err_msg)

        if collection_name is None:
            raise ValueError("Please provide a collection name.")
        # from chromadb.config import Settings

        if persist_directory is not None:
            self._client = chromadb.PersistentClient(
                path=persist_directory if persist_directory else "./chroma",
            )
        elif (host is not None) or (port is not None):
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
            )

        self._collection = self._client.get_collection(collection_name)

    def create_documents(self, results: Any) -> List[Document]:
        """Create documents from the results.

        Args:
            results: Results from the query.

        Returns:
            List of documents.
        """
        documents = []
        for result in zip(
            results["ids"],
            results["documents"],
            results["embeddings"],
            results["metadatas"],
        ):
            document = Document(
                id_=result[0][0],
                text=result[1][0],
                embedding=result[2][0],
                metadata=result[3][0],
            )
            documents.append(document)

        return documents

    def load_data(
        self,
        query_embedding: Optional[List[float]] = None,
        limit: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        query: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        """Load data from the collection.

        Args:
            limit: Number of results to return.
            where: Filter results by metadata. {"metadata_field": "is_equal_to_this"}
            where_document: Filter results by document. {"$contains":"search_string"}

        Returns:
            List of documents.
        """
        where = where or {}
        where_document = where_document or {}
        if query_embedding is not None:
            results = self._collection.search(
                query_embedding=query_embedding,
                n_results=limit,
                where=where,
                where_document=where_document,
                include=["metadatas", "documents", "distances", "embeddings"],
            )
            return self.create_documents(results)
        elif query is not None:
            query = query if isinstance(query, list) else [query]
            results = self._collection.query(
                query_texts=query,
                n_results=limit,
                where=where,
                where_document=where_document,
                include=["metadatas", "documents", "distances", "embeddings"],
            )
            return self.create_documents(results)
        else:
            raise ValueError("Please provide either query embedding or query.")
