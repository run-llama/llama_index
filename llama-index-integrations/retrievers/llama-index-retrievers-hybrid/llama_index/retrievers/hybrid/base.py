"""Hybrid Retriever combining dense and sparse retrieval."""

import logging
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


class FusionMode(str, Enum):
    """Fusion mode for combining retrieval results."""

    # Reciprocal Rank Fusion - works well when retrievers have different score scales
    RRF = "rrf"
    # Relative Score Fusion - normalizes scores to [0, 1] before combining
    RELATIVE_SCORE = "relative_score"
    # Distribution-Based Score Fusion - uses z-score normalization
    DIST_BASED_SCORE = "dist_based_score"
    # Simple weighted sum (requires similar score scales)
    WEIGHTED_SUM = "weighted_sum"


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever combining multiple retrievers with fusion strategies.

    This retriever combines results from multiple retrievers (typically dense/vector
    and sparse/BM25) using various fusion strategies like RRF, relative score fusion,
    or weighted sum.

    Args:
        retrievers: List of retrievers to combine.
        weights: Optional weights for each retriever. If not provided, equal weights
            are used. Must sum to 1.0 if provided.
        fusion_mode: The fusion strategy to use. Defaults to RRF.
        similarity_top_k: Number of results to return. Defaults to 10.
        rrf_k: The k parameter for RRF fusion. Higher values give more weight to
            lower-ranked results. Defaults to 60.
        callback_manager: Optional callback manager.
        verbose: Whether to log verbose output.

    Example:
        ```python
        from llama_index.retrievers.hybrid import HybridRetriever, FusionMode
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.retrievers.bm25 import BM25Retriever

        # Create individual retrievers
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
        bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=10)

        # Create hybrid retriever with RRF fusion
        hybrid_retriever = HybridRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4],
            fusion_mode=FusionMode.RRF,
            similarity_top_k=5,
        )

        # Retrieve
        nodes = hybrid_retriever.retrieve("What is LlamaIndex?")
        ```
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        fusion_mode: FusionMode = FusionMode.RRF,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        rrf_k: int = 60,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        if not retrievers:
            raise ValueError("At least one retriever must be provided.")

        self._retrievers = retrievers
        self._num_retrievers = len(retrievers)

        # Validate and set weights
        if weights is None:
            self._weights = [1.0 / self._num_retrievers] * self._num_retrievers
        else:
            if len(weights) != self._num_retrievers:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of retrievers ({self._num_retrievers})."
                )
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}.")
            self._weights = weights

        self._fusion_mode = fusion_mode
        self._similarity_top_k = similarity_top_k
        self._rrf_k = rrf_k

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @property
    def retrievers(self) -> List[BaseRetriever]:
        """Get the list of retrievers."""
        return self._retrievers

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from all retrievers and fuse results."""
        # Collect results from all retrievers
        all_results: List[List[NodeWithScore]] = []
        for retriever in self._retrievers:
            results = retriever.retrieve(query_bundle)
            all_results.append(results)

        # Fuse results based on fusion mode
        if self._fusion_mode == FusionMode.RRF:
            fused = self._rrf_fusion(all_results)
        elif self._fusion_mode == FusionMode.RELATIVE_SCORE:
            fused = self._relative_score_fusion(all_results)
        elif self._fusion_mode == FusionMode.DIST_BASED_SCORE:
            fused = self._dist_based_score_fusion(all_results)
        elif self._fusion_mode == FusionMode.WEIGHTED_SUM:
            fused = self._weighted_sum_fusion(all_results)
        else:
            raise ValueError(f"Unknown fusion mode: {self._fusion_mode}")

        # Sort by score and return top k
        fused.sort(key=lambda x: x.score or 0, reverse=True)
        return fused[: self._similarity_top_k]

    def _rrf_fusion(
        self, all_results: List[List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF score = sum(weight_i / (k + rank_i)) for each retriever i

        This method is robust to different score scales across retrievers.
        """
        # node_id -> (node, cumulative_score)
        node_scores: Dict[str, Tuple[NodeWithScore, float]] = {}

        for retriever_idx, results in enumerate(all_results):
            weight = self._weights[retriever_idx]
            for rank, node_with_score in enumerate(results, start=1):
                node_id = node_with_score.node.node_id
                rrf_score = weight / (self._rrf_k + rank)

                if node_id in node_scores:
                    existing_node, existing_score = node_scores[node_id]
                    node_scores[node_id] = (existing_node, existing_score + rrf_score)
                else:
                    node_scores[node_id] = (node_with_score, rrf_score)

        # Create final results with fused scores
        return [
            NodeWithScore(node=node.node, score=score)
            for node, score in node_scores.values()
        ]

    def _relative_score_fusion(
        self, all_results: List[List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """
        Relative Score Fusion.

        Normalizes scores to [0, 1] range within each retriever,
        then combines with weights.
        """
        node_scores: Dict[str, Tuple[NodeWithScore, float]] = {}

        for retriever_idx, results in enumerate(all_results):
            if not results:
                continue

            weight = self._weights[retriever_idx]

            # Get min and max scores for normalization
            scores = [r.score or 0 for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            for node_with_score in results:
                node_id = node_with_score.node.node_id
                raw_score = node_with_score.score or 0

                # Normalize to [0, 1]
                if score_range > 0:
                    normalized_score = (raw_score - min_score) / score_range
                else:
                    normalized_score = 1.0

                weighted_score = weight * normalized_score

                if node_id in node_scores:
                    existing_node, existing_score = node_scores[node_id]
                    node_scores[node_id] = (existing_node, existing_score + weighted_score)
                else:
                    node_scores[node_id] = (node_with_score, weighted_score)

        return [
            NodeWithScore(node=node.node, score=score)
            for node, score in node_scores.values()
        ]

    def _dist_based_score_fusion(
        self, all_results: List[List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """
        Distribution-Based Score Fusion.

        Uses z-score normalization based on mean and standard deviation,
        then combines with weights.
        """
        node_scores: Dict[str, Tuple[NodeWithScore, float]] = {}

        for retriever_idx, results in enumerate(all_results):
            if not results:
                continue

            weight = self._weights[retriever_idx]

            # Calculate mean and std
            scores = [r.score or 0 for r in results]
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_score = variance ** 0.5

            for node_with_score in results:
                node_id = node_with_score.node.node_id
                raw_score = node_with_score.score or 0

                # Z-score normalization
                if std_score > 0:
                    normalized_score = (raw_score - mean_score) / std_score
                else:
                    normalized_score = 0.0

                weighted_score = weight * normalized_score

                if node_id in node_scores:
                    existing_node, existing_score = node_scores[node_id]
                    node_scores[node_id] = (existing_node, existing_score + weighted_score)
                else:
                    node_scores[node_id] = (node_with_score, weighted_score)

        return [
            NodeWithScore(node=node.node, score=score)
            for node, score in node_scores.values()
        ]

    def _weighted_sum_fusion(
        self, all_results: List[List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """
        Simple Weighted Sum Fusion.

        Directly combines scores with weights. Best used when retrievers
        have similar score scales.
        """
        node_scores: Dict[str, Tuple[NodeWithScore, float]] = {}

        for retriever_idx, results in enumerate(all_results):
            weight = self._weights[retriever_idx]

            for node_with_score in results:
                node_id = node_with_score.node.node_id
                raw_score = node_with_score.score or 0
                weighted_score = weight * raw_score

                if node_id in node_scores:
                    existing_node, existing_score = node_scores[node_id]
                    node_scores[node_id] = (existing_node, existing_score + weighted_score)
                else:
                    node_scores[node_id] = (node_with_score, weighted_score)

        return [
            NodeWithScore(node=node.node, score=score)
            for node, score in node_scores.values()
        ]
