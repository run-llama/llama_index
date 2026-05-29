"""Retriever wrapping CockroachDBVectorStore with C-SPANN tuning."""

from __future__ import annotations

from typing import Any

from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore


class CockroachDBRetriever(BaseRetriever):
    """Standalone retriever over a CockroachDBVectorStore.

    Embeds the query with ``embed_model``, runs the vector search through the
    store, and returns ``NodeWithScore`` results. Use this when you want vector
    retrieval without going through ``VectorStoreIndex``, for example when
    inserting CRDB-side retrieval into a custom query pipeline.

    Examples:
        ```python
        retriever = CockroachDBRetriever(
            vector_store=store,
            embed_model="local",
            similarity_top_k=5,
            vector_search_beam_size=64,
        )
        nodes = retriever.retrieve("what is C-SPANN?")
        ```
    """

    def __init__(
        self,
        vector_store: CockroachDBVectorStore,
        embed_model: EmbedType | None = None,
        similarity_top_k: int = 4,
        mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: MetadataFilters | None = None,
        vector_search_beam_size: int | None = None,
        mmr_threshold: float | None = None,
        mmr_prefetch_factor: float | None = None,
        mmr_prefetch_k: int | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = resolve_embed_model(embed_model)
        self._similarity_top_k = similarity_top_k
        self._mode = mode
        self._filters = filters
        self._vector_search_beam_size = vector_search_beam_size
        self._mmr_threshold = mmr_threshold
        self._mmr_prefetch_factor = mmr_prefetch_factor
        self._mmr_prefetch_k = mmr_prefetch_k
        super().__init__(callback_manager=callback_manager)

    def _build_query_kwargs(self) -> dict:
        kwargs: dict = {}
        if self._vector_search_beam_size is not None:
            kwargs["vector_search_beam_size"] = self._vector_search_beam_size
        if self._mmr_threshold is not None:
            kwargs["mmr_threshold"] = self._mmr_threshold
        if self._mmr_prefetch_factor is not None:
            kwargs["mmr_prefetch_factor"] = self._mmr_prefetch_factor
        if self._mmr_prefetch_k is not None:
            kwargs["mmr_prefetch_k"] = self._mmr_prefetch_k
        return kwargs

    def _build_query(self, embedding: list[float]) -> VectorStoreQuery:
        return VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=self._similarity_top_k,
            filters=self._filters,
            mode=self._mode,
            mmr_threshold=self._mmr_threshold,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        embedding = (
            query_bundle.embedding
            if query_bundle.embedding is not None
            else self._embed_model.get_query_embedding(query_bundle.query_str)
        )
        result = self._vector_store.query(
            self._build_query(embedding), **self._build_query_kwargs()
        )
        return self._to_nodes_with_score(result)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        embedding = (
            query_bundle.embedding
            if query_bundle.embedding is not None
            else await self._embed_model.aget_query_embedding(query_bundle.query_str)
        )
        result = await self._vector_store.aquery(
            self._build_query(embedding), **self._build_query_kwargs()
        )
        return self._to_nodes_with_score(result)

    @staticmethod
    def _to_nodes_with_score(result: Any) -> list[NodeWithScore]:
        nodes = result.nodes or []
        sims = result.similarities or [None] * len(nodes)
        return [NodeWithScore(node=n, score=s) for n, s in zip(nodes, sims, strict=False)]
