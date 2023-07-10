from typing import Optional, Dict, List, Any

from llama_index.indices.base_retriever import BaseRetriever
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.indices.colbert.base import ColbertIndex
from llama_index.schema import NodeWithScore
from llama_index.indices.query.schema import QueryBundle
from llama_index.vector_stores.types import MetadataFilters

class ColbertRetriever(BaseRetriever):
    """Vector index retriever.

    Args:
        index (ColbertIndex): Colbert index.
        similarity_top_k (int): number of top k results to return.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        colbert_kwargs (dict): Additional colbert specific kwargs to pass
            through to the colbert index at query time.

    """

    def __init__(
        self,
        index: ColbertIndex,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        filters: Optional[MetadataFilters] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._service_context = self._index.service_context
        self._docstore = self._index.docstore

        self._similarity_top_k = similarity_top_k
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters

        self._kwargs: Dict[str, Any] = kwargs.get("colbert_kwargs", {})

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        return self._index.query(query_str=query_bundle.query_str, top_k=self._similarity_top_k, **self._kwargs)
