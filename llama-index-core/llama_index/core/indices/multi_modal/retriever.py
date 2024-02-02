"""Base vector store index query."""

import asyncio
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_multi_modal_retriever import (
    MultiModalRetriever,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.core.indices.utils import log_vector_store_query_result
from llama_index.core.schema import (
    NodeWithScore,
    ObjectType,
    QueryBundle,
    QueryType,
)
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)


class MultiModalVectorIndexRetriever(MultiModalRetriever):
    """Multi Modal Vector index retriever.

    Args:
        index (MultiModalVectorIndexRetriever): Multi Modal vector store index for images and texts.
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
        image_similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        sparse_top_k: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._vector_store = self._index.vector_store
        # separate image vector store for image retrieval
        self._image_vector_store = self._index.image_vector_store

        assert isinstance(self._index.image_embed_model, BaseEmbedding)
        self._image_embed_model = index._image_embed_model
        self._embed_model = index._embed_model

        self._service_context = self._index.service_context
        self._docstore = self._index.docstore

        self._similarity_top_k = similarity_top_k
        self._image_similarity_top_k = image_similarity_top_k
        self._vector_store_query_mode = VectorStoreQueryMode(vector_store_query_mode)
        self._alpha = alpha
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters
        self._sparse_top_k = sparse_top_k

        self._kwargs: Dict[str, Any] = kwargs.get("vector_store_kwargs", {})
        self.callback_manager = (
            callback_manager
            or callback_manager_from_settings_or_context(
                Settings, self._service_context
            )
        )

    @property
    def similarity_top_k(self) -> int:
        """Return similarity top k."""
        return self._similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, similarity_top_k: int) -> None:
        """Set similarity top k."""
        self._similarity_top_k = similarity_top_k

    @property
    def image_similarity_top_k(self) -> int:
        """Return image similarity top k."""
        return self._image_similarity_top_k

    @image_similarity_top_k.setter
    def image_similarity_top_k(self, image_similarity_top_k: int) -> None:
        """Set image similarity top k."""
        self._image_similarity_top_k = image_similarity_top_k

    def _build_vector_store_query(
        self, query_bundle_with_embeddings: QueryBundle, similarity_top_k: int
    ) -> VectorStoreQuery:
        return VectorStoreQuery(
            query_embedding=query_bundle_with_embeddings.embedding,
            similarity_top_k=similarity_top_k,
            node_ids=self._node_ids,
            doc_ids=self._doc_ids,
            query_str=query_bundle_with_embeddings.query_str,
            mode=self._vector_store_query_mode,
            alpha=self._alpha,
            filters=self._filters,
            sparse_top_k=self._sparse_top_k,
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        res = []
        # If text vector store is not empty, retrieve text nodes
        # If text vector store is empty, please create index without text vector store
        if self._vector_store is not None:
            res.extend(self._text_retrieve(query_bundle))

        # If image vector store is not empty, retrieve text nodes
        # If image vector store is empty, please create index without image vector store
        if self._image_vector_store is not None:
            res.extend(self._text_to_image_retrieve(query_bundle))
        return res

    def _text_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if not self._index.is_text_vector_store_empty:
            if self._vector_store.is_embedding_query:
                if (
                    query_bundle.embedding is None
                    and len(query_bundle.embedding_strs) > 0
                ):
                    query_bundle.embedding = (
                        self._embed_model.get_agg_embedding_from_queries(
                            query_bundle.embedding_strs
                        )
                    )
            return self._get_nodes_with_embeddings(
                query_bundle, self._similarity_top_k, self._vector_store
            )
        else:
            return []

    def text_retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._text_retrieve(str_or_query_bundle)

    def _text_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if not self._index.is_image_vector_store_empty:
            if self._image_vector_store.is_embedding_query:
                # change the embedding for query bundle to Multi Modal Text encoder
                query_bundle.embedding = (
                    self._image_embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
            return self._get_nodes_with_embeddings(
                query_bundle, self._image_similarity_top_k, self._image_vector_store
            )
        else:
            return []

    def text_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._text_to_image_retrieve(str_or_query_bundle)

    def _image_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if not self._index.is_image_vector_store_empty:
            if self._image_vector_store.is_embedding_query:
                # change the embedding for query bundle to Multi Modal Image encoder for image input
                assert isinstance(self._index.image_embed_model, MultiModalEmbedding)
                query_bundle.embedding = self._image_embed_model.get_image_embedding(
                    query_bundle.embedding_image[0]
                )
            return self._get_nodes_with_embeddings(
                query_bundle, self._image_similarity_top_k, self._image_vector_store
            )
        else:
            return []

    def image_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(
                query_str="", image_path=str_or_query_bundle
            )
        return self._image_to_image_retrieve(str_or_query_bundle)

    def _get_nodes_with_embeddings(
        self,
        query_bundle_with_embeddings: QueryBundle,
        similarity_top_k: int,
        vector_store: VectorStore,
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(
            query_bundle_with_embeddings, similarity_top_k
        )
        query_result = vector_store.query(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)

    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> List[NodeWithScore]:
        if query_result.nodes is None:
            # NOTE: vector store does not keep text and returns node indices.
            # Need to recover all nodes from docstore
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index.index_struct, IndexDict)
            node_ids = [
                self._index.index_struct.nodes_dict[idx] for idx in query_result.ids
            ]
            nodes = self._docstore.get_nodes(node_ids)
            query_result.nodes = nodes
        else:
            # NOTE: vector store keeps text, returns nodes.
            # Only need to recover image or index nodes from docstore
            for i in range(len(query_result.nodes)):
                source_node = query_result.nodes[i].source_node
                if (not self._vector_store.stores_text) or (
                    source_node is not None and source_node.node_type != ObjectType.TEXT
                ):
                    node_id = query_result.nodes[i].node_id
                    if self._docstore.document_exists(node_id):
                        query_result.nodes[
                            i
                        ] = self._docstore.get_node(  # type: ignore[index]
                            node_id
                        )

        log_vector_store_query_result(query_result)

        node_with_scores: List[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            node_with_scores.append(NodeWithScore(node=node, score=score))

        return node_with_scores

    # Async Retrieval Methods

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Run the two retrievals in async, and return their results as a concatenated list
        results: List[NodeWithScore] = []
        tasks = [
            self._atext_retrieve(query_bundle),
            self._atext_to_image_retrieve(query_bundle),
        ]

        task_results = await asyncio.gather(*tasks)

        for task_result in task_results:
            results.extend(task_result)
        return results

    async def _atext_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if not self._index.is_text_vector_store_empty:
            if self._vector_store.is_embedding_query:
                # change the embedding for query bundle to Multi Modal Text encoder
                query_bundle.embedding = (
                    await self._embed_model.aget_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
            return await self._aget_nodes_with_embeddings(
                query_bundle, self._similarity_top_k, self._vector_store
            )
        else:
            return []

    async def atext_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._atext_retrieve(str_or_query_bundle)

    async def _atext_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if not self._index.is_image_vector_store_empty:
            if self._image_vector_store.is_embedding_query:
                # change the embedding for query bundle to Multi Modal Text encoder
                query_bundle.embedding = (
                    await self._image_embed_model.aget_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
            return await self._aget_nodes_with_embeddings(
                query_bundle, self._image_similarity_top_k, self._image_vector_store
            )
        else:
            return []

    async def atext_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._atext_to_image_retrieve(str_or_query_bundle)

    async def _aget_nodes_with_embeddings(
        self,
        query_bundle_with_embeddings: QueryBundle,
        similarity_top_k: int,
        vector_store: VectorStore,
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(
            query_bundle_with_embeddings, similarity_top_k
        )
        query_result = await vector_store.aquery(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)

    async def _aimage_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if not self._index.is_image_vector_store_empty:
            if self._image_vector_store.is_embedding_query:
                # change the embedding for query bundle to Multi Modal Image encoder for image input
                assert isinstance(self._index.image_embed_model, MultiModalEmbedding)
                # Using the first imaage in the list for image retrieval
                query_bundle.embedding = (
                    await self._image_embed_model.aget_image_embedding(
                        query_bundle.embedding_image[0]
                    )
                )
            return await self._aget_nodes_with_embeddings(
                query_bundle, self._image_similarity_top_k, self._image_vector_store
            )
        else:
            return []

    async def aimage_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            # leave query_str as empty since we are using image_path for image retrieval
            str_or_query_bundle = QueryBundle(
                query_str="", image_path=str_or_query_bundle
            )
        return await self._aimage_to_image_retrieve(str_or_query_bundle)
