from typing import Any, List, cast, Optional, Literal, Dict


from llama_index.vector_stores.docarray.base import DocArrayVectorStore


class DocArrayInMemoryVectorStore(DocArrayVectorStore):
    def __init__(
        self,
        index_path: Optional[str] = None,
        metric: Literal[
            "cosine_sim", "euclidian_dist", "sgeuclidean_dist"
        ] = "cosine_sim",
    ):
        import_err_msg = """
                `docarray` package not found. Install the package via pip:
                `pip install docarray`
        """
        try:
            import docarray  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self._index_file_path = index_path
        self._ref_docs = None
        self._index, self._schema = self._init_index(metric=metric)

    def _init_index(self, **kwargs):
        from docarray.index import InMemoryExactNNIndex

        schema = self._get_schema(**kwargs)
        return (
            InMemoryExactNNIndex[schema](index_file_path=self._index_file_path),
            schema,
        )

    def _find_docs_to_be_removed(self, doc_id):
        query = {"metadata__doc_id": {"$eq": doc_id}}
        docs =  self._index.filter(query)
        return [doc.id for doc in docs]

    def persist(self, persist_path: str) -> None:
        index_path = persist_path or self._index_file_path
        self._index.persist(index_path)
