"""Hybrid FAISS Vector Store — dense + sparse Reciprocal Rank Fusion.

Extends FaissMapVectorStore with a BM25 sparse retrieval path, fused via
Reciprocal Rank Fusion (RRF).  No proprietary embedding service required.

Ported from kernel-ml/opensemanticsearch (Sathyavageeswaran, 2024).
Reference: US Patent 19/287,703.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import numpy as np

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.faiss.map_store import FaissMapVectorStore


class HybridFAISSVectorStore(FaissMapVectorStore):
    """Hybrid FAISS vector store combining dense similarity with BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge the two ranked lists:
    ``score(d) = dense_weight / (rrf_k + dense_rank) + bm25_weight / (rrf_k + sparse_rank)``

    Requires ``rank-bm25``: ``pip install rank-bm25``

    Args:
        faiss_index: A ``faiss.IndexIDMap`` or ``faiss.IndexIDMap2`` instance.
        bm25_weight: Weight for the BM25 sparse path (default 0.3).
        dense_weight: Weight for the FAISS dense path (default 0.7).
        rrf_k: RRF rank-smoothing constant (default 60).

    Example::

        import faiss
        from llama_index.vector_stores.faiss import HybridFAISSVectorStore

        index = faiss.IndexIDMap2(faiss.IndexFlatL2(1536))
        store = HybridFAISSVectorStore(faiss_index=index, bm25_weight=0.3)
        store.add(nodes)
        results = store.query(VectorStoreQuery(
            query_embedding=..., query_str="search keywords", similarity_top_k=5
        ))

    """

    bm25_weight: float = Field(default=0.3, description="Weight for the BM25 sparse path.")
    dense_weight: float = Field(default=0.7, description="Weight for the FAISS dense path.")
    rrf_k: int = Field(default=60, description="RRF rank-smoothing constant.")

    _bm25: Optional[Any] = PrivateAttr(default=None)
    _corpus_texts: List[str] = PrivateAttr(default_factory=list)
    _corpus_ids: List[str] = PrivateAttr(default_factory=list)

    def __init__(self, faiss_index: Any, bm25_weight: float = 0.3,
                 dense_weight: float = 0.7, rrf_k: int = 60) -> None:
        super().__init__(faiss_index=faiss_index)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Internal BM25 helpers
    # ------------------------------------------------------------------

    def _get_bm25(self) -> Any:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank-bm25 is required for HybridFAISSVectorStore. "
                "Install it with: pip install rank-bm25"
            ) from e
        if not self._corpus_texts:
            return None
        tokenized = [t.lower().split() for t in self._corpus_texts]
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized)

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current corpus."""
        self._bm25 = self._get_bm25()

    def _bm25_search(self, query_text: str, top_k: int) -> List[str]:
        """Return top_k node IDs ranked by BM25 score."""
        if not self._bm25 or not self._corpus_texts:
            return []
        tokens = query_text.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_idx[:top_k]:
            if idx < len(self._corpus_ids):
                results.append(self._corpus_ids[idx])
        return results

    def _reciprocal_rank_fusion(
        self, dense_ids: List[str], sparse_ids: List[str]
    ) -> List[tuple]:
        """Merge dense and sparse ranked lists via RRF.

        Returns list of (node_id, fused_score) sorted descending.
        """
        scores: Dict[str, float] = {}
        for rank, node_id in enumerate(dense_ids):
            scores[node_id] = scores.get(node_id, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)
        for rank, node_id in enumerate(sparse_ids):
            scores[node_id] = scores.get(node_id, 0.0) + self.bm25_weight / (self.rrf_k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes and update BM25 corpus."""
        node_ids = super().add(nodes, **add_kwargs)
        for node in nodes:
            text = node.get_content() or ""
            self._corpus_texts.append(text)
            self._corpus_ids.append(node.id_)
        self._rebuild_bm25()
        return node_ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete node and remove from BM25 corpus."""
        super().delete(ref_doc_id, **kwargs)
        if ref_doc_id in self._corpus_ids:
            idx = self._corpus_ids.index(ref_doc_id)
            self._corpus_texts.pop(idx)
            self._corpus_ids.pop(idx)
            self._rebuild_bm25()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Hybrid query: dense FAISS + BM25 sparse, fused via RRF.

        Falls back to pure dense retrieval when ``query.query_str`` is empty
        or BM25 corpus is not populated.
        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for Faiss yet.")

        top_k = query.similarity_top_k
        fetch_k = top_k * 3  # Over-fetch before fusion

        # Dense path
        query_embedding = cast(list, query.query_embedding)
        query_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        dists, indices = self._faiss_index.search(query_np, fetch_k)
        dense_ids = [
            self._faiss_id_to_node_id_map[int(idx)]
            for idx in indices[0]
            if idx >= 0 and int(idx) in self._faiss_id_to_node_id_map
        ]

        # Sparse path (only when query text is available)
        query_str = query.query_str or ""
        if query_str.strip() and self._bm25:
            sparse_ids = self._bm25_search(query_str, fetch_k)
        else:
            sparse_ids = []

        if not sparse_ids:
            # No BM25 data: return pure dense results
            dist_list = list(dists[0])
            filtered = [
                (float(d), self._faiss_id_to_node_id_map[int(i)])
                for d, i in zip(dist_list, indices[0])
                if i >= 0 and int(i) in self._faiss_id_to_node_id_map
            ][:top_k]
            sims, ids = zip(*filtered) if filtered else ([], [])
            return VectorStoreQueryResult(similarities=list(sims), ids=list(ids))

        # Fuse and return top-k
        fused = self._reciprocal_rank_fusion(dense_ids, sparse_ids)[:top_k]
        result_ids = [nid for nid, _ in fused]
        result_sims = [score for _, score in fused]
        return VectorStoreQueryResult(similarities=result_sims, ids=result_ids)
