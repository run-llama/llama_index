"""Integration tests for WhooshRetriever (requires llama-index-core + whoosh3)."""

from __future__ import annotations

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.whoosh import WhooshRetriever

TEXTS = [
    "Whoosh is a fast pure-Python full-text search library.",
    "BM25 ranks documents by term rarity and frequency.",
    "Elasticsearch is a distributed search engine written in Java.",
]
IDS = ["a", "b", "c"]
METADATAS = [{"src": "readme"}, {"src": "docs"}, {"src": "wiki"}]


def _retriever(k: int = 4) -> WhooshRetriever:
    return WhooshRetriever.from_texts(texts=TEXTS, ids=IDS, metadatas=METADATAS, k=k)


def test_is_base_retriever():
    assert isinstance(_retriever(), BaseRetriever)


def test_retrieve_returns_nodes():
    nodes = _retriever().retrieve("pure python search")
    assert nodes, "expected at least one hit"
    assert all(isinstance(n, NodeWithScore) for n in nodes)
    # The exact-token query should float the Whoosh readme doc to the top.
    assert nodes[0].node.metadata["id"] == "a"


def test_metadata_is_preserved_and_scored():
    nodes = _retriever().retrieve("term rarity")
    top = nodes[0]
    assert top.node.metadata["id"] == "b"
    assert top.node.metadata["src"] == "docs"
    assert isinstance(top.score, float)
    assert top.score > 0


def test_k_limits_results():
    nodes = _retriever(k=1).retrieve("search")
    assert len(nodes) == 1


def test_empty_query_returns_nothing():
    assert _retriever().retrieve("   ") == []


def test_from_index_roundtrip(tmp_path):
    WhooshRetriever.from_texts(
        texts=TEXTS, ids=IDS, metadatas=METADATAS, path=str(tmp_path), k=4
    )
    reopened = WhooshRetriever.from_index(str(tmp_path), k=4)
    nodes = reopened.retrieve("distributed java engine")
    assert nodes[0].node.metadata["id"] == "c"
