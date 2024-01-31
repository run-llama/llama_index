from typing import Any, List, Literal, Optional

import fsspec

from llama_index.vector_stores.docarray.base import DocArrayVectorStore


class DocArrayInMemoryVectorStore(DocArrayVectorStore):
    """Class representing a DocArray In-Memory vector store.

    This class is a document index provided by Docarray that stores documents in memory.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        metric: Literal[
            "cosine_sim", "euclidian_dist", "sgeuclidean_dist"
        ] = "cosine_sim",
    ):
        """Initializes the DocArrayInMemoryVectorStore.

        Args:
            index_path (Optional[str]): The path to the index file.
            metric (Literal["cosine_sim", "euclidian_dist", "sgeuclidean_dist"]):
                The distance metric to use. Default is "cosine_sim".
        """
        import_err_msg = """
                `docarray` package not found. Install the package via pip:
                `pip install docarray`
        """
        try:
            import docarray  # noqa
        except ImportError:
            raise ImportError(import_err_msg)

        self._ref_docs = None  # type: ignore[assignment]
        self._index_file_path = index_path
        self._index, self._schema = self._init_index(metric=metric)

    def _init_index(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        """Initializes the in-memory exact nearest neighbour index.

        Args:
            **kwargs: Variable length argument list.

        Returns:
            tuple: The in-memory exact nearest neighbour index and its schema.
        """
        from docarray.index import InMemoryExactNNIndex

        schema = self._get_schema(**kwargs)
        index = InMemoryExactNNIndex[schema]  # type: ignore[valid-type]
        params = {"index_file_path": self._index_file_path}
        return index(**params), schema  # type: ignore[arg-type]

    def _find_docs_to_be_removed(self, doc_id: str) -> List[str]:
        """Finds the documents to be removed from the vector store.

        Args:
            doc_id (str): Reference document ID that should be removed.

        Returns:
            List[str]: List of document IDs to be removed.
        """
        query = {"metadata__doc_id": {"$eq": doc_id}}
        docs = self._index.filter(query)
        return [doc.id for doc in docs]

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persists the in-memory vector store to a file.

        Args:
            persist_path (str): The path to persist the index.
            fs (fsspec.AbstractFileSystem, optional): Filesystem to persist to.
                (doesn't apply)
        """
        index_path = persist_path or self._index_file_path
        self._index.persist(index_path)
