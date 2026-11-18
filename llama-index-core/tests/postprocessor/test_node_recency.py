"""Test the recency postprocessors."""

from typing import Any, List
from unittest.mock import patch

import pytest

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.postprocessor.node_recency import EmbeddingRecencyPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

pytest.importorskip("pandas", reason="pandas is required for recency postprocessors")

# "alpha" and "gamma" have identical embeddings, "beta" is orthogonal to both.
VECTORS = {
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "gamma": [1.0, 0.0],
}


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    return VECTORS[text]


def mock_get_query_embedding(text: str) -> List[float]:
    """Mock get query embedding."""
    return VECTORS[text]


def make_node(text: str, node_id: str, date: str) -> TextNode:
    """Create a node with a date in its metadata."""
    return TextNode(
        text=text,
        id_=node_id,
        metadata={"date": date},
        excluded_embed_metadata_keys=["date"],
    )


@patch.object(MockEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding)
@patch.object(
    MockEmbedding, "_get_query_embedding", side_effect=mock_get_query_embedding
)
def test_embedding_recency_postprocessor_uses_matching_embeddings(
    _mock_query: Any, _mock_text: Any
) -> None:
    """
    Each node must be compared against its own embedding.

    The nodes are not passed in date order, so the postprocessor has to sort them
    before looking up embeddings. "alpha" (oldest) has the same embedding as
    "gamma" (newer), so "alpha" must be dropped, while "gamma" must be kept
    because it is not similar to the most recent node "beta".
    """
    nodes = [
        make_node("alpha", "alpha", "2020-01-01"),
        make_node("beta", "beta", "2020-01-03"),
        make_node("gamma", "gamma", "2020-01-02"),
    ]
    nodes_with_scores = [NodeWithScore(node=node) for node in nodes]

    postprocessor = EmbeddingRecencyPostprocessor(
        embed_model=MockEmbedding(embed_dim=2),
        query_embedding_tmpl="{context_str}",
    )
    result_nodes = postprocessor.postprocess_nodes(
        nodes_with_scores, query_bundle=QueryBundle(query_str="What is?")
    )

    assert [node.node.node_id for node in result_nodes] == ["beta", "gamma"]
