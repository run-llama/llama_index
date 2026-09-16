"""Tests for EmbeddingRecencyPostprocessor dedup (no network/LLM)."""

from typing import List

import pytest

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor.node_recency import (
    EmbeddingRecencyPostprocessor,
)
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode

pandas = pytest.importorskip("pandas")


class _ContentEmbedding(BaseEmbedding):
    """Deterministic, content-dependent embedding: alpha->[1,0], beta->[0,1]."""

    @classmethod
    def class_name(cls) -> str:
        return "_ContentEmbedding"

    def _vec(self, text: str) -> List[float]:
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._vec(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._vec(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._vec(text)


def test_embedding_recency_dedup_keeps_unique_when_input_not_date_sorted() -> None:
    # Input order differs from descending-date order (the normal retriever case).
    # Two 'alpha' duplicates and one unique 'beta'; the older alpha should be
    # dropped and beta kept. Previously the embedding list was built in input
    # order but indexed by date-sorted position, so beta was dropped instead.
    nodes = [
        NodeWithScore(
            node=TextNode(
                id_="n0",
                text="alpha",
                metadata={"date": "2020-01-01"},
                excluded_embed_metadata_keys=["date"],
            )
        ),
        NodeWithScore(
            node=TextNode(
                id_="n1",
                text="alpha",
                metadata={"date": "2020-01-03"},
                excluded_embed_metadata_keys=["date"],
            )
        ),
        NodeWithScore(
            node=TextNode(
                id_="n2",
                text="beta",
                metadata={"date": "2020-01-02"},
                excluded_embed_metadata_keys=["date"],
            )
        ),
    ]

    postprocessor = EmbeddingRecencyPostprocessor(
        embed_model=_ContentEmbedding(), similarity_cutoff=0.5
    )
    result = postprocessor.postprocess_nodes(
        nodes, query_bundle=QueryBundle(query_str="q")
    )

    contents = sorted(n.node.get_content(metadata_mode=MetadataMode.NONE) for n in result)
    # The unique 'beta' must survive; exactly one 'alpha' is kept.
    assert contents == ["alpha", "beta"]
    kept_ids = {n.node.node_id for n in result}
    assert "n2" in kept_ids  # beta kept
    assert "n1" in kept_ids  # newest alpha kept
    assert "n0" not in kept_ids  # older alpha dropped
