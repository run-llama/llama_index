"""Tests for Hybrid Retriever."""

import pytest
from typing import List
from unittest.mock import MagicMock, patch

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from llama_index.retrievers.hybrid import HybridRetriever, FusionMode


def create_mock_retriever(results: List[NodeWithScore]) -> BaseRetriever:
    """Create a mock retriever that returns specified results."""
    mock = MagicMock(spec=BaseRetriever)
    mock.retrieve.return_value = results
    return mock


def create_node_with_score(node_id: str, text: str, score: float) -> NodeWithScore:
    """Create a NodeWithScore for testing."""
    node = TextNode(id_=node_id, text=text)
    return NodeWithScore(node=node, score=score)


class TestHybridRetrieverInit:
    """Tests for HybridRetriever initialization."""

    def test_init_with_single_retriever(self) -> None:
        """Test initialization with a single retriever."""
        mock_retriever = create_mock_retriever([])
        hybrid = HybridRetriever(retrievers=[mock_retriever])

        assert len(hybrid.retrievers) == 1
        assert hybrid._weights == [1.0]

    def test_init_with_multiple_retrievers_default_weights(self) -> None:
        """Test initialization with multiple retrievers and default weights."""
        mock1 = create_mock_retriever([])
        mock2 = create_mock_retriever([])
        hybrid = HybridRetriever(retrievers=[mock1, mock2])

        assert len(hybrid.retrievers) == 2
        assert hybrid._weights == [0.5, 0.5]

    def test_init_with_custom_weights(self) -> None:
        """Test initialization with custom weights."""
        mock1 = create_mock_retriever([])
        mock2 = create_mock_retriever([])
        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            weights=[0.7, 0.3],
        )

        assert hybrid._weights == [0.7, 0.3]

    def test_init_empty_retrievers_raises(self) -> None:
        """Test that empty retrievers list raises ValueError."""
        with pytest.raises(ValueError, match="At least one retriever"):
            HybridRetriever(retrievers=[])

    def test_init_mismatched_weights_raises(self) -> None:
        """Test that mismatched weights count raises ValueError."""
        mock = create_mock_retriever([])
        with pytest.raises(ValueError, match="Number of weights"):
            HybridRetriever(retrievers=[mock], weights=[0.5, 0.5])

    def test_init_weights_not_sum_to_one_raises(self) -> None:
        """Test that weights not summing to 1 raises ValueError."""
        mock1 = create_mock_retriever([])
        mock2 = create_mock_retriever([])
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            HybridRetriever(retrievers=[mock1, mock2], weights=[0.6, 0.6])


class TestRRFFusion:
    """Tests for RRF fusion mode."""

    def test_rrf_fusion_basic(self) -> None:
        """Test basic RRF fusion with two retrievers."""
        # Create results for two retrievers
        results1 = [
            create_node_with_score("node1", "text1", 0.9),
            create_node_with_score("node2", "text2", 0.8),
        ]
        results2 = [
            create_node_with_score("node2", "text2", 0.95),
            create_node_with_score("node3", "text3", 0.7),
        ]

        mock1 = create_mock_retriever(results1)
        mock2 = create_mock_retriever(results2)

        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            fusion_mode=FusionMode.RRF,
            similarity_top_k=10,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        # node2 should have highest score (appears in both)
        node_ids = [r.node.node_id for r in results]
        assert "node2" in node_ids
        assert "node1" in node_ids
        assert "node3" in node_ids

        # node2 should be ranked highest
        assert results[0].node.node_id == "node2"

    def test_rrf_fusion_with_weights(self) -> None:
        """Test RRF fusion with custom weights."""
        results1 = [create_node_with_score("node1", "text1", 0.9)]
        results2 = [create_node_with_score("node2", "text2", 0.95)]

        mock1 = create_mock_retriever(results1)
        mock2 = create_mock_retriever(results2)

        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            weights=[0.8, 0.2],
            fusion_mode=FusionMode.RRF,
            similarity_top_k=10,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        # node1 should be ranked higher due to weight
        assert results[0].node.node_id == "node1"


class TestRelativeScoreFusion:
    """Tests for relative score fusion mode."""

    def test_relative_score_fusion_basic(self) -> None:
        """Test basic relative score fusion."""
        results1 = [
            create_node_with_score("node1", "text1", 100),
            create_node_with_score("node2", "text2", 50),
        ]
        results2 = [
            create_node_with_score("node2", "text2", 0.9),
            create_node_with_score("node3", "text3", 0.5),
        ]

        mock1 = create_mock_retriever(results1)
        mock2 = create_mock_retriever(results2)

        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            fusion_mode=FusionMode.RELATIVE_SCORE,
            similarity_top_k=10,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        # All nodes should be present
        node_ids = [r.node.node_id for r in results]
        assert len(node_ids) == 3

    def test_relative_score_fusion_single_result(self) -> None:
        """Test relative score fusion with single result per retriever."""
        results1 = [create_node_with_score("node1", "text1", 1.0)]
        results2 = [create_node_with_score("node2", "text2", 0.5)]

        mock1 = create_mock_retriever(results1)
        mock2 = create_mock_retriever(results2)

        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            fusion_mode=FusionMode.RELATIVE_SCORE,
            similarity_top_k=10,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        # Both nodes should have normalized score of 1.0 (only one result each)
        assert len(results) == 2


class TestDistBasedScoreFusion:
    """Tests for distribution-based score fusion mode."""

    def test_dist_based_score_fusion_basic(self) -> None:
        """Test basic distribution-based score fusion."""
        results1 = [
            create_node_with_score("node1", "text1", 10),
            create_node_with_score("node2", "text2", 8),
            create_node_with_score("node3", "text3", 6),
        ]
        results2 = [
            create_node_with_score("node2", "text2", 0.9),
            create_node_with_score("node4", "text4", 0.7),
        ]

        mock1 = create_mock_retriever(results1)
        mock2 = create_mock_retriever(results2)

        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            fusion_mode=FusionMode.DIST_BASED_SCORE,
            similarity_top_k=10,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        # All unique nodes should be present
        node_ids = [r.node.node_id for r in results]
        assert len(node_ids) == 4


class TestWeightedSumFusion:
    """Tests for weighted sum fusion mode."""

    def test_weighted_sum_fusion_basic(self) -> None:
        """Test basic weighted sum fusion."""
        results1 = [create_node_with_score("node1", "text1", 0.8)]
        results2 = [create_node_with_score("node1", "text1", 0.6)]

        mock1 = create_mock_retriever(results1)
        mock2 = create_mock_retriever(results2)

        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            weights=[0.5, 0.5],
            fusion_mode=FusionMode.WEIGHTED_SUM,
            similarity_top_k=10,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        # Expected score: 0.5 * 0.8 + 0.5 * 0.6 = 0.7
        assert len(results) == 1
        assert abs(results[0].score - 0.7) < 1e-6


class TestTopKLimit:
    """Tests for top_k limiting."""

    def test_top_k_limits_results(self) -> None:
        """Test that results are limited to top_k."""
        results1 = [
            create_node_with_score(f"node{i}", f"text{i}", 1.0 - i * 0.1)
            for i in range(10)
        ]
        mock = create_mock_retriever(results1)

        hybrid = HybridRetriever(
            retrievers=[mock],
            fusion_mode=FusionMode.RRF,
            similarity_top_k=3,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        assert len(results) == 3


class TestEmptyResults:
    """Tests for handling empty results."""

    def test_empty_results_from_one_retriever(self) -> None:
        """Test handling when one retriever returns empty results."""
        results1 = [create_node_with_score("node1", "text1", 0.9)]
        results2: List[NodeWithScore] = []

        mock1 = create_mock_retriever(results1)
        mock2 = create_mock_retriever(results2)

        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            fusion_mode=FusionMode.RRF,
            similarity_top_k=10,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        assert len(results) == 1
        assert results[0].node.node_id == "node1"

    def test_all_empty_results(self) -> None:
        """Test handling when all retrievers return empty results."""
        mock1 = create_mock_retriever([])
        mock2 = create_mock_retriever([])

        hybrid = HybridRetriever(
            retrievers=[mock1, mock2],
            fusion_mode=FusionMode.RRF,
            similarity_top_k=10,
        )

        query = QueryBundle(query_str="test query")
        results = hybrid._retrieve(query)

        assert len(results) == 0
