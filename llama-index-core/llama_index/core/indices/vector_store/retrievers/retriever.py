"""Base vector store index query."""

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.utils import log_vector_store_query_result
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, ObjectType, QueryBundle
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class VectorIndexRetriever(BaseRetriever):
    """
    Vector index retriever.

    Args:
        index (VectorStoreIndex): vector store index.
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
        index: VectorStoreIndex,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        sparse_top_k: Optional[int] = None,
        hybrid_top_k: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        embed_model: Optional[BaseEmbedding] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._vector_store = self._index.vector_store
        self._embed_model = embed_model or self._index._embed_model
        self._docstore = self._index.docstore

        self._similarity_top_k = similarity_top_k
        self._vector_store_query_mode = VectorStoreQueryMode(vector_store_query_mode)
        self._alpha = alpha
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters
        self._sparse_top_k = sparse_top_k
        self._hybrid_top_k = hybrid_top_k
        self._kwargs: Dict[str, Any] = kwargs.get("vector_store_kwargs", {})

        callback_manager = callback_manager or CallbackManager()
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            verbose=verbose,
        )

    @property
    def similarity_top_k(self) -> int:
        """Return similarity top k."""
        return self._similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, similarity_top_k: int) -> None:
        """Set similarity top k."""
        self._similarity_top_k = similarity_top_k

    def _needs_embedding(self) -> bool:
        """Check if the current query mode requires embeddings."""
        return (
            self._vector_store.is_embedding_query
            and self._vector_store_query_mode
            not in (
                VectorStoreQueryMode.TEXT_SEARCH,
                VectorStoreQueryMode.SPARSE,
            )
        )

    @dispatcher.span
    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._needs_embedding():
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                query_bundle.embedding = (
                    self._embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
        return self._get_nodes_with_embeddings(query_bundle)

    @dispatcher.span
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        embedding = query_bundle.embedding
        if self._needs_embedding():
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                embed_model = self._embed_model
                embedding = await embed_model.aget_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
        return await self._aget_nodes_with_embeddings(
            QueryBundle(query_str=query_bundle.query_str, embedding=embedding)
        )

    def _build_vector_store_query(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> VectorStoreQuery:
        return VectorStoreQuery(
            query_embedding=query_bundle_with_embeddings.embedding,
            similarity_top_k=self._similarity_top_k,
            node_ids=self._node_ids,
            doc_ids=self._doc_ids,
            query_str=query_bundle_with_embeddings.query_str,
            mode=self._vector_store_query_mode,
            alpha=self._alpha,
            filters=self._filters,
            sparse_top_k=self._sparse_top_k,
            hybrid_top_k=self._hybrid_top_k,
        )

    def _determine_nodes_to_fetch(
        self, query_result: VectorStoreQueryResult
    ) -> list[str]:
        """
        Determine the nodes to fetch from the docstore.

        If the vector store does not store text, we need to fetch every node from the docstore.
        If the vector store stores text, we need to fetch only the nodes that are not text.
        """
        if query_result.nodes:
            # Fetch non-text nodes from the docstore
            return [
                node.node_id
                for node in query_result.nodes  # no folding
                if node.as_related_node_info().node_type
                != ObjectType.TEXT  # TODO: no need to fetch multimodal `Node` if they only include text
            ]
        elif query_result.ids:
            # Fetch all nodes from the docstore
            return [
                self._index.index_struct.nodes_dict[idx] for idx in query_result.ids
            ]

        else:
            return []

    def _insert_fetched_nodes_into_query_result(
        self, query_result: VectorStoreQueryResult, fetched_nodes: List[BaseNode]
    ) -> Sequence[BaseNode]:
        """
        Insert the fetched nodes into the query result.

        If the vector store does not store text, all nodes are inserted into the query result.
        If the vector store stores text, we replace non-text nodes with those fetched from the docstore,
            unless the node was not found in the docstore, in which case we keep the original node.
        """
        fetched_nodes_by_id: Dict[str, BaseNode] = {
            str(node.node_id): node for node in fetched_nodes
        }
        new_nodes: List[BaseNode] = []

        if query_result.nodes:
            for node in list(query_result.nodes):
                node_id_str = str(node.node_id)
                if node_id_str in fetched_nodes_by_id:
                    new_nodes.append(fetched_nodes_by_id[node_id_str])
                else:
                    # We did not fetch a replacement node, so we keep the original node
                    new_nodes.append(node)
        elif query_result.ids:
            for node_id in query_result.ids:
                if node_id not in self._index.index_struct.nodes_dict:
                    raise KeyError(f"Node ID {node_id} not found in index. ")
                node_id_str = str(self._index.index_struct.nodes_dict[node_id])
                if node_id_str in fetched_nodes_by_id:
                    new_nodes.append(fetched_nodes_by_id[node_id_str])
                else:
                    raise KeyError(
                        f"Node ID {node_id_str} not found in fetched nodes. "
                    )
        elif query_result.ids is None and query_result.nodes is None:
            raise ValueError(
                "Vector store query result should return at least one of nodes or ids."
            )
        return new_nodes

    def _convert_nodes_to_scored_nodes(
        self, query_result: VectorStoreQueryResult
    ) -> List[NodeWithScore]:
        """Create scored nodes from the vector store query result."""
        node_with_scores: List[NodeWithScore] = []

        for ind, node in enumerate(list(query_result.nodes or [])):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]

            node_with_scores.append(NodeWithScore(node=node, score=score))

        return node_with_scores

    def _get_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = self._vector_store.query(query, **self._kwargs)

        nodes_to_fetch = self._determine_nodes_to_fetch(query_result)
        if nodes_to_fetch:
            # Fetch any missing nodes from the docstore and insert them into the query result
            fetched_nodes: List[BaseNode] = self._docstore.get_nodes(
                node_ids=nodes_to_fetch, raise_error=False
            )

            query_result.nodes = self._insert_fetched_nodes_into_query_result(
                query_result, fetched_nodes
            )

        log_vector_store_query_result(query_result)

        return self._convert_nodes_to_scored_nodes(query_result)

    async def _aget_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = await self._vector_store.aquery(query, **self._kwargs)

        nodes_to_fetch = self._determine_nodes_to_fetch(query_result)
        if nodes_to_fetch:
            # Fetch any missing nodes from the docstore and insert them into the query result
            fetched_nodes: List[BaseNode] = await self._docstore.aget_nodes(
                node_ids=nodes_to_fetch, raise_error=False
            )

            query_result.nodes = self._insert_fetched_nodes_into_query_result(
                query_result, fetched_nodes
            )

        log_vector_store_query_result(query_result)

        return self._convert_nodes_to_scored_nodes(query_result)
