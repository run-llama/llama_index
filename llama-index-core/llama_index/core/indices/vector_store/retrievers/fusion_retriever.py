from __future__ import annotations

import asyncio
from typing import List, Dict, Optional, Iterable

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode


def _mode_requires_embedding(mode: VectorStoreQueryMode) -> bool:
    """Return True only for modes that need dense embeddings."""
    return mode in {
        VectorStoreQueryMode.DEFAULT,
        VectorStoreQueryMode.MMR,
        VectorStoreQueryMode.HYBRID,
        VectorStoreQueryMode.SEMANTIC_HYBRID,
    }


def _retriever_needs_embedding(r: BaseRetriever) -> bool:
    vs = getattr(r, "_vector_store", None)
    mode = getattr(r, "_vector_store_query_mode", VectorStoreQueryMode.DEFAULT)
    return bool(vs and getattr(vs, "is_embedding_query", False) and _mode_requires_embedding(mode))


def _first_embed_model(retrievers: Iterable[BaseRetriever]) -> Optional[BaseEmbedding]:
    for r in retrievers:
        em = getattr(r, "_embed_model", None)
        if em is not None:
            return em
    return None


class QueryFusionRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        top_k: int = 10,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self._retrievers = retrievers
        self._top_k = top_k

        # either explicitly given
        self._embed_model = embed_model

        # or try to steal from a child retriever
        if self._embed_model is None:
            for r in self._retrievers:
                child_model = getattr(r, "_embed_model", None)
                if child_model is not None:
                    self._embed_model = child_model
                    break

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        qb = self._prepare_qb_sync(query_bundle)
        combined: List[NodeWithScore] = []
        for r in self._retrievers:
            results = r._retrieve(qb)
            if results is None:
                continue
            combined.extend(results)

        combined.sort(
            key=lambda x: (x.score if x.score is not None else 0.0),
            reverse=True,
        )
        return combined[: self._top_k]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        qb = await self._prepare_qb_async(query_bundle)

        tasks = [r._aretrieve(qb) for r in self._retrievers]
        groups = await asyncio.gather(*tasks)

        combined: List[NodeWithScore] = []
        for g in groups:
            if g is None:
                continue
            combined.extend(g)

        combined.sort(
            key=lambda x: (x.score if x.score is not None else 0.0),
            reverse=True,
        )
        return combined[:self._top_k]
    def _prepare_qb_sync(self, qb: QueryBundle) -> QueryBundle:
        if not any(_retriever_needs_embedding(r) for r in self._retrievers):
            return qb

        if qb.embedding is not None or not getattr(qb, "embedding_strs", []):
            return qb

        if self._embed_model is None:
            raise RuntimeError("No embed model available in fusion (sync).")

        embedding = self._embed_model.get_agg_embedding_from_queries(qb.embedding_strs)
        return QueryBundle(query_str=qb.query_str, embedding=embedding)



    async def _prepare_qb_async(self, qb: QueryBundle) -> QueryBundle:
        if not any(_retriever_needs_embedding(r) for r in self._retrievers):
            return qb

        if qb.embedding is not None or not getattr(qb, "embedding_strs", []):
            return qb

        if self._embed_model is None:
            raise RuntimeError("No embed model available in fusion (async).")

        embedding = await self._embed_model.aget_agg_embedding_from_queries(
            qb.embedding_strs
        )
        return QueryBundle(query_str=qb.query_str, embedding=embedding)
      

    # ---------- simple merge policy ----------

    @staticmethod
    def _merge(nodes: List[NodeWithScore], top_k: Optional[int]) -> List[NodeWithScore]:
        # dedupe by node_id, keep best score, then sort desc by score
        best: Dict[str, NodeWithScore] = {}
        for n in nodes:
            nid = str(getattr(n.node, "node_id", ""))
            prev = best.get(nid)
            if prev is None or (
                n.score is not None and (prev.score is None or n.score > prev.score)
            ):
                best[nid] = n
        out = list(best.values())
        out.sort(key=lambda x: (x.score is not None, x.score), reverse=True)
        return out[:top_k] if top_k is not None else out