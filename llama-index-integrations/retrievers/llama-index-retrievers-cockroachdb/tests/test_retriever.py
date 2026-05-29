"""Tests for CockroachDBRetriever (uses a stub embedder, no model downloads)."""

from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode

from llama_index.retrievers.cockroachdb import CockroachDBRetriever
from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore

EMBED_DIM = 4


class HashEmbedding(BaseEmbedding):
    """Deterministic 4-d embedding so tests don't hit the network."""

    def _get_query_embedding(self, query: str) -> list[float]:
        h = hash(query) % 1000 / 1000.0
        return [h, 1.0, 1.0, 1.0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)


@pytest.fixture()
def populated_store(fresh_db: dict[str, Any]) -> CockroachDBVectorStore:
    s = CockroachDBVectorStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        database=fresh_db["database"],
        table_name="retr_idx",
        embed_dim=EMBED_DIM,
        sslmode="disable",
    )
    embed = HashEmbedding()
    nodes = []
    for nid, txt in [("a", "alpha"), ("b", "beta"), ("c", "gamma")]:
        n = TextNode(id_=nid, text=txt)
        n.embedding = embed._get_text_embedding(txt)
        nodes.append(n)
    s.add(nodes)
    return s


def test_retriever_returns_nodes_with_scores(populated_store: CockroachDBVectorStore) -> None:
    retr = CockroachDBRetriever(
        vector_store=populated_store,
        embed_model=HashEmbedding(),
        similarity_top_k=2,
    )
    out = retr.retrieve("alpha")
    assert len(out) == 2
    assert all(n.node.id_ in {"a", "b", "c"} for n in out)
    assert out[0].score is not None
