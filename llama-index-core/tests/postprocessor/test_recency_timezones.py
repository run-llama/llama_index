import pytest

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.postprocessor.node_recency import (
    EmbeddingRecencyPostprocessor,
    FixedRecencyPostprocessor,
)
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.mark.parametrize("embedding_recency", [False, True])
def test_recency_orders_mixed_utc_offsets_by_instant(embedding_recency: bool) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(
                id_="older",
                text="Older report",
                metadata={"date": "2026-01-01T01:00:00+01:00"},
            )
        ),
        NodeWithScore(
            node=TextNode(
                id_="newer",
                text="Newer report",
                metadata={"date": "2026-01-01T00:30:00+00:00"},
            )
        ),
    ]
    if embedding_recency:
        processor = EmbeddingRecencyPostprocessor(
            embed_model=MockEmbedding(embed_dim=2), similarity_cutoff=1.1
        )
    else:
        processor = FixedRecencyPostprocessor(top_k=2)

    result = processor.postprocess_nodes(nodes, query_str="Recent reports")

    assert [node.node.node_id for node in result] == ["newer", "older"]
