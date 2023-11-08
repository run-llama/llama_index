"""Base vector store index query."""


from typing import Any, Dict, List, Optional

from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.schema import NodeWithScore
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStoreQueryMode,
)


class MutliModalVectorIndexRetriever(VectorIndexRetriever):
    """Multi Modal Vector index retriever.

    Args:
        index (MutliModalVectorIndexRetriever): Multi Modal vector store index for images and texts.
        similarity_top_k (int): number of top k results to return.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        alpha (float): weight for sparse/dense retrieval, only used for
            hybrid query mode.
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

    """

    def __init__(
        self,
        index: MultiModalVectorStoreIndex,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        sparse_top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._vector_store = self._index.vector_store
        # separate image vector store for image retrieval
        self._image_vector_store = self._index.image_vector_store
        self._image_embed_model = self._index.image_embed_model
        self._service_context = self._index.service_context
        self._docstore = self._index.docstore

        self._similarity_top_k = similarity_top_k
        self._vector_store_query_mode = VectorStoreQueryMode(vector_store_query_mode)
        self._alpha = alpha
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters
        self._sparse_top_k = sparse_top_k

        self._kwargs: Dict[str, Any] = kwargs.get("vector_store_kwargs", {})

    @property
    def similarity_top_k(self) -> int:
        """Return similarity top k."""
        return self._similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, similarity_top_k: int) -> None:
        """Set similarity top k."""
        self._similarity_top_k = similarity_top_k

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        res = self._text_retrieve(query_bundle)
        res.extend(self._image_retrieve(query_bundle))
        return res

    async def _aretrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        res = self._atext_retrieve(query_bundle)
        res.extend(self._aimage_retrieve(query_bundle))
        return res

    def _text_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                # get text embedding from Text embed model
                query_bundle.embedding = (
                    self._service_context.embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
        return self._get_nodes_with_embeddings(query_bundle)

    async def _atext_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                # get text embedding from Text embed model
                query_bundle.embedding = await self._service_context.embed_model.aget_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
        return self._aget_nodes_with_embeddings(query_bundle)

    def _image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._image_vector_store.is_embedding_query:
            # change the embedding for query bundle to Multi Modal Text encoder
            query_bundle.embedding = (
                self._image_embed_model.get_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )
        return self._get_image_nodes_with_image_embeddings(query_bundle)

    async def _aimage_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._image_vector_store.is_embedding_query:
            # change the embedding for query bundle to Multi Modal Text encoder
            query_bundle.embedding = (
                await self._image_embed_model.aget_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )
        return self._aget_image_nodes_with_image_embeddings(query_bundle)

    # for image nodes retrieval
    def _get_image_nodes_with_image_embeddings(
        self,
        query_bundle_with_embeddings: QueryBundle,
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = self._image_vector_store.query(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)

    async def _aget_image_nodes_with_image_embeddings(
        self,
        query_bundle_with_embeddings: QueryBundle,
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = await self._image_vector_store.aquery(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)
