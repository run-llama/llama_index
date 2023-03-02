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
        host: str = "localhost",
        port: int = 8000,
        persist_directory: str = None,
    ) -> None:

        """Initialize with parameters."""
        import_err_msg = "`chromadb` package not found, please run `pip install chromadb`"
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        if collection_name is None:
            raise ValueError("Please provide a collection name.")
        from chromadb.config import Settings

        if persist_directory:
            self._client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
        else:
            self._client = chromadb.Client(
                Settings(
                    chroma_api_impl="rest",
                    chroma_server_host=host,
                    chroma_server_http_port=port,
                )
            )
        self._collection = self._client.get_collection(collection_name)

    def load_data(
        self,
        query: str or list,
        limit: int = 10,
        where: dict = {},  # {"metadata_field": "is_equal_to_this"},
        where_document: dict = {},  # {"$contains":"search_string"}
    ) -> Any:
        print(self._collection.count())
        query = query if isinstance(query, list) else [query]
        results = self._collection.query(
            query_texts=query,
            n_results=limit,
            where=where,
            where_document=where_document,
            include=["metadatas", "documents", "distances", "embeddings"],
        )
        print(results)
        documents = []
        for result in zip(
            results["ids"],
            results["documents"],
            results["embeddings"],
            results["metadatas"],
        ):
            document = Document(
                doc_id=result[0][0],
                text=result[1][0],
                embedding=result[2][0],
                extra_info=result[3][0],
            )
            documents.append(document)

        return documents
