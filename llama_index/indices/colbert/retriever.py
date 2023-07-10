from llama_index.indices.base_retriever import BaseRetriever


class ColbertRetriever(BaseRetriever):
    """Vector index retriever.

    Args:
        index (VectorStoreIndex): vector store index.
        similarity_top_k (int): number of top k results to return.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

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
        self._vector_store = self._index.vector_store
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
        return self._index.query(query, top_k=self._similarity_top_k, **self._kwargs)
