"""Tests for AutoMergingRetriever gap-filling logic."""

from unittest.mock import MagicMock, patch

from llama_index.core.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    RelatedNodeInfo,
    TextNode,
)


def _build_linked_nodes(texts):
    """Build a list of TextNode objects linked via PREVIOUS / NEXT."""
    nodes = [TextNode(text=t, id_=f"node_{i}") for i, t in enumerate(texts)]
    for i in range(len(nodes) - 1):
        nodes[i].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=nodes[i + 1].node_id
        )
        nodes[i + 1].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=nodes[i].node_id
        )
    return nodes


def _make_retriever(nodes):
    """Return an AutoMergingRetriever backed by a mock docstore."""
    docstore = MagicMock()
    docstore.get_document = MagicMock(
        side_effect=lambda nid: next(n for n in nodes if n.node_id == nid)
    )
    storage = MagicMock()
    storage.docstore = docstore
    vector_retriever = MagicMock()
    return AutoMergingRetriever(vector_retriever, storage)


class TestFillInNodes:
    """Tests for _fill_in_nodes gap-filling."""

    def test_forward_order_fills_gap(self):
        """Gap is detected when nodes arrive in document order."""
        nodes = _build_linked_nodes(["A", "B", "C"])
        retriever = _make_retriever(nodes)

        # Simulate retrieval returning node_0 and node_2 (forward order)
        initial = [
            NodeWithScore(node=nodes[0], score=0.9),
            NodeWithScore(node=nodes[2], score=0.8),
        ]

        result, changed = retriever._fill_in_nodes(initial)

        assert changed is True
        result_ids = [n.node.node_id for n in result]
        assert result_ids == ["node_0", "node_1", "node_2"]

    def test_reverse_order_fills_gap(self):
        """Gap is detected when nodes arrive in reverse document order.

        This is the core bug from issue #24262: when the vector store
        returns [node_2, node_0] the gap node_1 was not filled.
        """
        nodes = _build_linked_nodes(["A", "B", "C"])
        retriever = _make_retriever(nodes)

        # Simulate retrieval returning node_2 before node_0 (reverse order)
        initial = [
            NodeWithScore(node=nodes[2], score=0.95),
            NodeWithScore(node=nodes[0], score=0.85),
        ]

        result, changed = retriever._fill_in_nodes(initial)

        assert changed is True
        result_ids = [n.node.node_id for n in result]
        # The gap node (node_1) should appear between the two hits
        assert "node_1" in result_ids
        assert len(result_ids) == 3

    def test_no_gap_no_change(self):
        """Adjacent nodes (no gap) should not trigger filling."""
        nodes = _build_linked_nodes(["A", "B", "C"])
        retriever = _make_retriever(nodes)

        initial = [
            NodeWithScore(node=nodes[0], score=0.9),
            NodeWithScore(node=nodes[1], score=0.8),
        ]

        result, changed = retriever._fill_in_nodes(initial)

        assert changed is False
        assert len(result) == 2

    def test_unrelated_nodes_no_change(self):
        """Nodes from different chains should not trigger filling."""
        chain_a = _build_linked_nodes(["A1", "A2"])
        chain_b = _build_linked_nodes(["B1", "B2"])
        # Give chain_b distinct ids
        for i, n in enumerate(chain_b):
            n.id_ = f"other_{i}"
            # Re-link with updated ids
        chain_b[0].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=chain_b[1].node_id
        )
        chain_b[1].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=chain_b[0].node_id
        )

        all_nodes = chain_a + chain_b
        retriever = _make_retriever(all_nodes)

        initial = [
            NodeWithScore(node=chain_a[0], score=0.9),
            NodeWithScore(node=chain_b[1], score=0.8),
        ]

        result, changed = retriever._fill_in_nodes(initial)

        assert changed is False
        assert len(result) == 2

    def test_gap_score_is_average(self):
        """The filled-in node should have the average score of its neighbours."""
        nodes = _build_linked_nodes(["A", "B", "C"])
        retriever = _make_retriever(nodes)

        initial = [
            NodeWithScore(node=nodes[2], score=0.9),
            NodeWithScore(node=nodes[0], score=0.7),
        ]

        result, _ = retriever._fill_in_nodes(initial)

        gap = [n for n in result if n.node.node_id == "node_1"][0]
        assert gap.score == (0.9 + 0.7) / 2
