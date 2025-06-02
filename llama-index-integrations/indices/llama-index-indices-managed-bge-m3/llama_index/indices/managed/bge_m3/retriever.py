from typing import Any, Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.indices.managed.bge_m3.base import BGEM3Index


class BGEM3Retriever(BaseRetriever):
    """
    Vector index retriever.

    Args:
        index (BGEM3Index): BGEM3 index.
        similarity_top_k (int): number of top k results to return.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        bge_m3_kwargs (dict): Additional bge_m3 specific kwargs to pass
            through to the bge_m3 index at query time.

    """

    def __init__(
        self,
        index: BGEM3Index,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        filters: Optional[MetadataFilters] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._docstore = self._index.docstore
        self._similarity_top_k = similarity_top_k
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters
        self._kwargs: Dict[str, Any] = kwargs.get("bge_m3_kwargs", {})
        self._model = self._index.model
        self._batch_size = self._index.batch_size
        self._query_maxlen = self._index.query_maxlen
        self._weights_for_different_modes = self._index.weights_for_different_modes
        super().__init__(
            callback_manager=callback_manager or Settings.callback_manager,
            object_map=object_map,
            verbose=verbose,
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        return self._index.query(
            query_str=query_bundle.query_str,
            top_k=self._similarity_top_k,
            **self._kwargs,
        )
